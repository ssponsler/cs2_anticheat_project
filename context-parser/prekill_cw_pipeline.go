package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
)

type pipelineArgs struct {
	demosRoot      string
	jsonlOut       string
	csvOut         string
	includeSeq     bool
	includeUnknown bool
	continueOnFail bool
	keepInterim    bool
	verbose        bool
}

type demoLabel struct {
	ID   int
	Name string
}

func parsePipelineArgs() pipelineArgs {
	demosRoot := flag.String("demos-root", "./test/cs-demos", "Root folder containing scraped .dem files")
	jsonlOut := flag.String("jsonl-out", "./context-parser/out/prekill_cw_dataset.jsonl", "Path to merged JSONL dataset")
	csvOut := flag.String("csv-out", "./context-parser/out/prekill_cw_dataset.csv", "Path to merged CSV dataset")
	includeSeq := flag.Bool("include-seq", false, "Include per-tick sequence channels in extractor JSONL output")
	includeUnknown := flag.Bool("include-unknown", false, "Include demos whose label cannot be inferred from path")
	continueOnFail := flag.Bool("continue-on-fail", true, "Continue processing remaining demos when one demo fails")
	keepInterim := flag.Bool("keep-interim", false, "Keep per-demo temporary extraction files")
	verbose := flag.Bool("verbose", true, "Print per-demo progress details")
	flag.Parse()

	return pipelineArgs{
		demosRoot:      *demosRoot,
		jsonlOut:       *jsonlOut,
		csvOut:         *csvOut,
		includeSeq:     *includeSeq,
		includeUnknown: *includeUnknown,
		continueOnFail: *continueOnFail,
		keepInterim:    *keepInterim,
		verbose:        *verbose,
	}
}

func main() {
	args := parsePipelineArgs()

	repoRoot, err := findRepoRoot()
	check(err)

	demosRootAbs := resolvePath(repoRoot, args.demosRoot)
	jsonlOutAbs := resolvePath(repoRoot, args.jsonlOut)
	csvOutAbs := resolvePath(repoRoot, args.csvOut)

	demoPaths, err := listDemoFiles(demosRootAbs)
	check(err)

	if len(demoPaths) == 0 {
		panic(fmt.Sprintf("no .dem files found under %s", demosRootAbs))
	}

	check(os.MkdirAll(filepath.Dir(jsonlOutAbs), 0o755))
	check(os.MkdirAll(filepath.Dir(csvOutAbs), 0o755))

	jsonlFile, err := os.Create(jsonlOutAbs)
	check(err)
	defer func() { check(jsonlFile.Close()) }()

	csvFile, err := os.Create(csvOutAbs)
	check(err)
	defer func() { check(csvFile.Close()) }()

	mergedCSV := csv.NewWriter(csvFile)
	defer func() {
		mergedCSV.Flush()
		check(mergedCSV.Error())
	}()

	interimDir := filepath.Join(repoRoot, "context-parser", "out", ".prekill_interim")
	check(os.MkdirAll(interimDir, 0o755))
	if !args.keepInterim {
		defer os.RemoveAll(interimDir)
	}

	if args.verbose {
		fmt.Printf("Discovered %d demo(s) under %s\n", len(demoPaths), demosRootAbs)
	}

	csvHeaderWritten := false
	processedDemos := 0
	skippedDemos := 0
	failedDemos := 0
	totalRows := 0

	for i, demoPath := range demoPaths {
		label := inferLabel(demoPath)
		if label.ID < 0 && !args.includeUnknown {
			skippedDemos++
			if args.verbose {
				fmt.Printf("[%d/%d] SKIP unlabeled demo: %s\n", i+1, len(demoPaths), demoPath)
			}
			continue
		}

		demoBase := strings.TrimSuffix(filepath.Base(demoPath), filepath.Ext(demoPath))
		tempJSONL := filepath.Join(interimDir, demoBase+".jsonl")
		tempCSV := filepath.Join(interimDir, demoBase+".csv")

		if args.verbose {
			fmt.Printf("[%d/%d] Processing %s | label=%s\n", i+1, len(demoPaths), demoPath, label.Name)
		}

		err := runPrekillExtractor(repoRoot, demoPath, tempJSONL, tempCSV, args.includeSeq)
		if err != nil {
			failedDemos++
			msg := fmt.Sprintf("extractor failed for %s: %v", demoPath, err)
			if args.continueOnFail {
				fmt.Println("WARN:", msg)
				continue
			}
			panic(msg)
		}

		rowsWritten, err := mergeJSONL(tempJSONL, jsonlFile, demoPath, label)
		if err != nil {
			failedDemos++
			msg := fmt.Sprintf("failed to merge JSONL for %s: %v", demoPath, err)
			if args.continueOnFail {
				fmt.Println("WARN:", msg)
				continue
			}
			panic(msg)
		}

		csvHeaderWritten, err = mergeCSV(tempCSV, mergedCSV, demoPath, label, csvHeaderWritten)
		if err != nil {
			failedDemos++
			msg := fmt.Sprintf("failed to merge CSV for %s: %v", demoPath, err)
			if args.continueOnFail {
				fmt.Println("WARN:", msg)
				continue
			}
			panic(msg)
		}

		processedDemos++
		totalRows += rowsWritten
	}

	fmt.Printf("\nPipeline complete\n")
	fmt.Printf("Processed demos: %d\n", processedDemos)
	fmt.Printf("Skipped demos:   %d\n", skippedDemos)
	fmt.Printf("Failed demos:    %d\n", failedDemos)
	fmt.Printf("Total rows:      %d\n", totalRows)
	fmt.Printf("JSONL output:    %s\n", jsonlOutAbs)
	fmt.Printf("CSV output:      %s\n", csvOutAbs)
}

func runPrekillExtractor(repoRoot, demoPath, jsonlOut, csvOut string, includeSeq bool) error {
	includeSeqArg := "false"
	if includeSeq {
		includeSeqArg = "true"
	}

	cmd := exec.Command(
		"go",
		"run",
		"./context-parser/prekill_cw.go",
		"-demo", demoPath,
		"-out-format", "both",
		"-jsonl-out", jsonlOut,
		"-csv-out", csvOut,
		"-include-seq", includeSeqArg,
	)
	cmd.Dir = repoRoot
	cmd.Stdout = io.Discard
	cmd.Stderr = os.Stderr
	if err := cmd.Run(); err != nil {
		return err
	}
	return nil
}

func mergeJSONL(inputPath string, mergedFile *os.File, demoPath string, label demoLabel) (int, error) {
	in, err := os.Open(inputPath)
	if err != nil {
		return 0, err
	}
	defer in.Close()

	reader := bufio.NewReader(in)
	writer := bufio.NewWriter(mergedFile)
	defer writer.Flush()

	rows := 0
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			var rec map[string]any
			if unmarshalErr := json.Unmarshal(line, &rec); unmarshalErr != nil {
				return rows, fmt.Errorf("invalid JSONL record in %s: %w", inputPath, unmarshalErr)
			}
			rec["demo_path"] = demoPath
			rec["demo_name"] = filepath.Base(demoPath)
			rec["label"] = label.ID
			rec["label_name"] = label.Name

			encoded, marshalErr := json.Marshal(rec)
			if marshalErr != nil {
				return rows, marshalErr
			}
			if _, writeErr := writer.Write(encoded); writeErr != nil {
				return rows, writeErr
			}
			if writeErr := writer.WriteByte('\n'); writeErr != nil {
				return rows, writeErr
			}
			rows++
		}

		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return rows, err
		}
	}

	if err := writer.Flush(); err != nil {
		return rows, err
	}
	return rows, nil
}

func mergeCSV(
	inputPath string,
	merged *csv.Writer,
	demoPath string,
	label demoLabel,
	headerWritten bool,
) (bool, error) {
	in, err := os.Open(inputPath)
	if err != nil {
		return headerWritten, err
	}
	defer in.Close()

	r := csv.NewReader(in)
	rows, err := r.ReadAll()
	if err != nil {
		return headerWritten, err
	}
	if len(rows) == 0 {
		return headerWritten, nil
	}

	header := rows[0]
	if !headerWritten {
		prefixed := append([]string{"demo_path", "demo_name", "label", "label_name"}, header...)
		if err := merged.Write(prefixed); err != nil {
			return headerWritten, err
		}
		headerWritten = true
	}

	for _, row := range rows[1:] {
		prefixed := append([]string{demoPath, filepath.Base(demoPath), fmt.Sprintf("%d", label.ID), label.Name}, row...)
		if err := merged.Write(prefixed); err != nil {
			return headerWritten, err
		}
	}

	merged.Flush()
	if err := merged.Error(); err != nil {
		return headerWritten, err
	}

	return headerWritten, nil
}

func listDemoFiles(root string) ([]string, error) {
	var files []string
	err := filepath.WalkDir(root, func(path string, d os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		if strings.EqualFold(filepath.Ext(d.Name()), ".dem") {
			files = append(files, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	sort.Strings(files)
	return files, nil
}

func inferLabel(path string) demoLabel {
	normalized := strings.ToLower(filepath.ToSlash(path))

	positiveTokens := []string{"/cheater/", "/cheaters/", "/hacker/", "/hackers/", "/rage/", "/spinbot/"}
	negativeTokens := []string{"/no-cheater/", "/no_cheater/", "/no-cheaters/", "/no_cheaters/", "/legit/", "/clean/", "/non_cheater/", "/non-cheater/"}

	for _, t := range negativeTokens {
		if strings.Contains(normalized, t) {
			return demoLabel{ID: 0, Name: "no_cheater"}
		}
	}
	for _, t := range positiveTokens {
		if strings.Contains(normalized, t) {
			return demoLabel{ID: 1, Name: "cheater"}
		}
	}

	return demoLabel{ID: -1, Name: "unknown"}
}

func findRepoRoot() (string, error) {
	cwd, err := os.Getwd()
	if err != nil {
		return "", err
	}

	cur := cwd
	for {
		if _, err := os.Stat(filepath.Join(cur, "go.mod")); err == nil {
			return cur, nil
		}
		next := filepath.Dir(cur)
		if next == cur {
			return "", errors.New("could not locate repository root (go.mod not found)")
		}
		cur = next
	}
}

func resolvePath(repoRoot, p string) string {
	if filepath.IsAbs(p) {
		return filepath.Clean(p)
	}
	return filepath.Clean(filepath.Join(repoRoot, p))
}

func check(err error) {
	if err != nil {
		panic(err)
	}
}
