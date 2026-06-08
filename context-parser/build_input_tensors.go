package main

import (
	"archive/zip"
	"bufio"
	"bytes"
	"encoding/binary"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

const (
	tensorSchemaVersion = "player_kill_window_tensor_v1"
)

type tensorArgs struct {
	inJSONL     string
	outJSONL    string
	outNPYDir   string
	outNPZPath  string
	seqLen      int
	keepUnknown bool
	exportNPY   bool
	exportNPZ   bool
}

type pipelineRecord struct {
	DemoPath                  string    `json:"demo_path"`
	DemoName                  string    `json:"demo_name"`
	Map                       string    `json:"map"`
	AvgRankRaw                string    `json:"avg_rank_raw"`
	Tick                      int       `json:"tick"`
	Round                     int       `json:"round"`
	KillerSteamID64           uint64    `json:"killer_steamid64"`
	VictimSteamID64           uint64    `json:"victim_steamid64"`
	Weapon                    string    `json:"weapon"`
	IsHeadshot                bool      `json:"is_headshot"`
	ThroughSmoke              bool      `json:"through_smoke"`
	PenetratedObjects         int       `json:"penetrated_objects"`
	KillDistance              float64   `json:"kill_distance"`
	WindowTicks               int       `json:"window_ticks"`
	PctVictimSpotted          float64   `json:"pct_victim_spotted"`
	AngularOffsetAtKill       float64   `json:"angular_offset_at_kill_deg"`
	MinHiddenAngularOffset    float64   `json:"min_hidden_angular_offset_deg"`
	MaxHiddenAngularOffset    float64   `json:"max_hidden_angular_offset_deg"`
	HiddenTrackingSlope       float64   `json:"hidden_tracking_slope_deg_per_tick"`
	VictimAudibleFootsteps    int       `json:"victim_audible_footsteps"`
	VictimAudibleWeaponFire   int       `json:"victim_audible_weapon_fire"`
	VictimAudibleJumps        int       `json:"victim_audible_jumps"`
	TicksSinceLastVictimSound int       `json:"ticks_since_last_victim_sound"`
	KillerAvgSpeed            float64   `json:"killer_avg_speed_units_per_tick"`
	KillerScopedAtKill        bool      `json:"killer_scoped_at_kill"`
	Label                     int       `json:"label"`
	LabelName                 string    `json:"label_name"`
	SeqAngularOffsetDeg       []float64 `json:"seq_ang_offset_deg"`
	SeqSpottedFlag            []uint8   `json:"seq_spotted_flag"`
	SeqKillerSpeed            []float64 `json:"seq_killer_speed"`
	SeqKillerScopedFlag       []uint8   `json:"seq_killer_scoped_flag"`
	SeqVictimAudibleAnyFlag   []uint8   `json:"seq_victim_audible_any_flag"`
}

type tensorSample struct {
	SchemaVersion      string      `json:"schema_version"`
	SampleID           string      `json:"sample_id"`
	DemoName           string      `json:"demo_name"`
	DemoPath           string      `json:"demo_path"`
	KillTick           int         `json:"kill_tick"`
	Round              int         `json:"round"`
	KillerSteamID64    uint64      `json:"killer_steamid64"`
	VictimSteamID64    uint64      `json:"victim_steamid64"`
	TargetLabel        int         `json:"target_label"`
	TargetLabelName    string      `json:"target_label_name"`
	XSeq               [][]float32 `json:"x_seq"`
	XSeqMask           []uint8     `json:"x_seq_mask"`
	XGlobal            []float32   `json:"x_global"`
	SeqFeatureNames    []string    `json:"seq_feature_names"`
	GlobalFeatureNames []string    `json:"global_feature_names"`
}

func parseTensorArgs() tensorArgs {
	inJSONL := flag.String("in-jsonl", "./context-parser/out/prekill_cw_dataset.jsonl", "Input merged prekill JSONL")
	outJSONL := flag.String("out-jsonl", "./context-parser/out/player_kill_window_tensor_v1.jsonl", "Output tensor JSONL")
	outNPYDir := flag.String("out-npy-dir", "", "Output directory for NumPy .npy exports (defaults to sibling folder when -export-npy=true)")
	outNPZPath := flag.String("out-npz", "", "Output path for convenience .npz bundle (defaults under NPY dir when -export-npz=true)")
	seqLen := flag.Int("seq-len", 192, "Fixed sequence length T for X_seq")
	keepUnknown := flag.Bool("keep-unknown", false, "Keep rows with unknown label (-1)")
	exportNPY := flag.Bool("export-npy", true, "Export NumPy .npy arrays for direct model training")
	exportNPZ := flag.Bool("export-npz", true, "Export convenience .npz bundle for direct model training")
	flag.Parse()

	if *seqLen <= 0 {
		panic("-seq-len must be > 0")
	}

	return tensorArgs{
		inJSONL:     *inJSONL,
		outJSONL:    *outJSONL,
		outNPYDir:   *outNPYDir,
		outNPZPath:  *outNPZPath,
		seqLen:      *seqLen,
		keepUnknown: *keepUnknown,
		exportNPY:   *exportNPY,
		exportNPZ:   *exportNPZ,
	}
}

func main() {
	args := parseTensorArgs()

	repoRoot, err := findRepoRoot()
	check(err)

	inPath := resolvePath(repoRoot, args.inJSONL)
	outPath := resolvePath(repoRoot, args.outJSONL)
	npyDirPath := ""
	npzPath := ""
	if args.exportNPY || args.exportNPZ {
		if args.outNPYDir != "" {
			npyDirPath = resolvePath(repoRoot, args.outNPYDir)
		} else {
			npyDirPath = filepath.Join(filepath.Dir(outPath), "player_kill_window_tensor_v1_npy")
		}
	}
	if args.exportNPZ {
		if args.outNPZPath != "" {
			npzPath = resolvePath(repoRoot, args.outNPZPath)
		} else {
			npzPath = filepath.Join(npyDirPath, "player_kill_window_tensor_v1.npz")
		}
	}

	inFile, err := os.Open(inPath)
	check(err)
	defer func() { check(inFile.Close()) }()

	check(os.MkdirAll(filepath.Dir(outPath), 0o755))
	outFile, err := os.Create(outPath)
	check(err)
	defer func() { check(outFile.Close()) }()

	scanner := bufio.NewScanner(inFile)
	scanner.Buffer(make([]byte, 1024*1024), 16*1024*1024)

	writer := bufio.NewWriter(outFile)
	defer func() {
		check(writer.Flush())
	}()

	processed := 0
	written := 0
	skippedUnknown := 0
	skippedMissingSeq := 0
	skippedMalformed := 0

	collector := newTensorCollector(args.seqLen)
	mapEncoder := newCategoricalEncoder("map")
	weaponEncoder := newCategoricalEncoder("weapon")

	for scanner.Scan() {
		processed++

		var rec pipelineRecord
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			skippedMalformed++
			continue
		}

		if rec.Label < 0 && !args.keepUnknown {
			skippedUnknown++
			continue
		}

		mapID := mapEncoder.Encode(rec.Map)
		weaponID := weaponEncoder.Encode(rec.Weapon)
		avgRankNumeric, hasAvgRank := parseAvgRankNumeric(rec.AvgRankRaw)

		tensor, ok := buildTensorSample(rec, args.seqLen, mapID, weaponID, avgRankNumeric, hasAvgRank)
		if !ok {
			skippedMissingSeq++
			continue
		}

		line, err := json.Marshal(tensor)
		if err != nil {
			panic(err)
		}
		if _, err := writer.Write(line); err != nil {
			panic(err)
		}
		if err := writer.WriteByte('\n'); err != nil {
			panic(err)
		}
		written++
		if args.exportNPY || args.exportNPZ {
			if err := collector.Append(tensor); err != nil {
				panic(err)
			}
		}
	}
	check(scanner.Err())

	collector.mapVocab = mapEncoder.Export()
	collector.weaponVocab = weaponEncoder.Export()

	if args.exportNPY {
		check(collector.Write(npyDirPath))
	}
	if args.exportNPZ {
		check(collector.WriteNPZ(npzPath))
	}

	fmt.Printf("Tensor build complete\n")
	fmt.Printf("Input rows:              %d\n", processed)
	fmt.Printf("Tensor rows written:     %d\n", written)
	fmt.Printf("Skipped unknown labels:  %d\n", skippedUnknown)
	fmt.Printf("Skipped missing seq:     %d\n", skippedMissingSeq)
	fmt.Printf("Skipped malformed rows:  %d\n", skippedMalformed)
	fmt.Printf("Output path:             %s\n", outPath)
	if args.exportNPY {
		fmt.Printf("NumPy output dir:        %s\n", npyDirPath)
	}
	if args.exportNPZ {
		fmt.Printf("NumPy NPZ bundle:        %s\n", npzPath)
	}
}

type tensorCollector struct {
	seqLen             int
	seqFeatures        int
	globalFeatures     int
	seqFeatureNames    []string
	globalFeatureNames []string
	mapVocab           map[string]int
	weaponVocab        map[string]int
	sampleIDs          []string
	xSeq               []float32
	xGlobal            []float32
	xSeqMask           []uint8
	y                  []int64
}

type categoricalEncoder struct {
	name   string
	toID   map[string]int
	nextID int
}

var rankNumberPattern = regexp.MustCompile(`[-+]?\d*\.?\d+`)

func newCategoricalEncoder(name string) *categoricalEncoder {
	return &categoricalEncoder{
		name:   name,
		toID:   map[string]int{"<unknown>": 0},
		nextID: 1,
	}
}

func (e *categoricalEncoder) Encode(value string) int {
	normalized := normalizeCategory(value)
	if normalized == "" {
		return 0
	}
	id, ok := e.toID[normalized]
	if ok {
		return id
	}
	id = e.nextID
	e.toID[normalized] = id
	e.nextID++
	return id
}

func (e *categoricalEncoder) Export() map[string]int {
	out := make(map[string]int, len(e.toID))
	for k, v := range e.toID {
		out[k] = v
	}
	return out
}

func normalizeCategory(v string) string {
	v = strings.TrimSpace(strings.ToLower(v))
	if v == "" || v == "unknown" || v == "null" || v == "n/a" {
		return ""
	}
	return v
}

func parseAvgRankNumeric(raw string) (float64, bool) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, false
	}
	match := rankNumberPattern.FindString(raw)
	if match == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(match, 64)
	if err != nil {
		return 0, false
	}
	return v, true
}

func newTensorCollector(seqLen int) *tensorCollector {
	return &tensorCollector{seqLen: seqLen}
}

func (c *tensorCollector) Append(sample tensorSample) error {
	if len(sample.XSeq) != c.seqLen {
		return fmt.Errorf("invalid sequence length for sample %s: got %d want %d", sample.SampleID, len(sample.XSeq), c.seqLen)
	}

	if c.seqFeatures == 0 {
		if len(sample.XSeq) == 0 {
			return fmt.Errorf("empty x_seq for sample %s", sample.SampleID)
		}
		c.seqFeatures = len(sample.XSeq[0])
		c.seqFeatureNames = append([]string(nil), sample.SeqFeatureNames...)
	}
	if c.globalFeatures == 0 {
		c.globalFeatures = len(sample.XGlobal)
		c.globalFeatureNames = append([]string(nil), sample.GlobalFeatureNames...)
	}

	if len(sample.XGlobal) != c.globalFeatures {
		return fmt.Errorf("invalid global feature size for sample %s: got %d want %d", sample.SampleID, len(sample.XGlobal), c.globalFeatures)
	}
	if len(sample.XSeqMask) != c.seqLen {
		return fmt.Errorf("invalid sequence mask length for sample %s: got %d want %d", sample.SampleID, len(sample.XSeqMask), c.seqLen)
	}

	for i := 0; i < c.seqLen; i++ {
		if len(sample.XSeq[i]) != c.seqFeatures {
			return fmt.Errorf("invalid sequence feature size at t=%d for sample %s: got %d want %d", i, sample.SampleID, len(sample.XSeq[i]), c.seqFeatures)
		}
		c.xSeq = append(c.xSeq, sample.XSeq[i]...)
	}

	c.xGlobal = append(c.xGlobal, sample.XGlobal...)
	c.xSeqMask = append(c.xSeqMask, sample.XSeqMask...)
	c.y = append(c.y, int64(sample.TargetLabel))
	c.sampleIDs = append(c.sampleIDs, sample.SampleID)

	return nil
}

func (c *tensorCollector) Write(outDir string) error {
	if outDir == "" {
		return errors.New("output numpy directory path is empty")
	}
	if c.seqFeatures == 0 {
		return errors.New("cannot write NumPy outputs: no tensor rows collected")
	}

	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return err
	}

	n := len(c.y)
	if err := writeNPYFloat32(filepath.Join(outDir, "x_seq.npy"), []int{n, c.seqLen, c.seqFeatures}, c.xSeq); err != nil {
		return err
	}
	if err := writeNPYFloat32(filepath.Join(outDir, "x_global.npy"), []int{n, c.globalFeatures}, c.xGlobal); err != nil {
		return err
	}
	if err := writeNPYUInt8(filepath.Join(outDir, "x_seq_mask.npy"), []int{n, c.seqLen}, c.xSeqMask); err != nil {
		return err
	}
	if err := writeNPYInt64(filepath.Join(outDir, "y.npy"), []int{n}, c.y); err != nil {
		return err
	}

	if err := writeJSON(filepath.Join(outDir, "sample_ids.json"), c.sampleIDs); err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(outDir, "map_vocab.json"), sortedVocab(c.mapVocab)); err != nil {
		return err
	}
	if err := writeJSON(filepath.Join(outDir, "weapon_vocab.json"), sortedVocab(c.weaponVocab)); err != nil {
		return err
	}
	meta := map[string]any{
		"schema_version":       tensorSchemaVersion,
		"num_samples":          n,
		"x_seq_shape":          []int{n, c.seqLen, c.seqFeatures},
		"x_global_shape":       []int{n, c.globalFeatures},
		"x_seq_mask_shape":     []int{n, c.seqLen},
		"y_shape":              []int{n},
		"seq_feature_names":    c.seqFeatureNames,
		"global_feature_names": c.globalFeatureNames,
		"map_vocab":            sortedVocab(c.mapVocab),
		"weapon_vocab":         sortedVocab(c.weaponVocab),
		"numpy_files": map[string]string{
			"x_seq":        "x_seq.npy",
			"x_global":     "x_global.npy",
			"x_seq_mask":   "x_seq_mask.npy",
			"y":            "y.npy",
			"map_vocab":    "map_vocab.json",
			"weapon_vocab": "weapon_vocab.json",
		},
	}
	if err := writeJSON(filepath.Join(outDir, "metadata.json"), meta); err != nil {
		return err
	}

	return nil
}

func (c *tensorCollector) WriteNPZ(outPath string) error {
	if outPath == "" {
		return errors.New("output npz path is empty")
	}
	if c.seqFeatures == 0 {
		return errors.New("cannot write NPZ output: no tensor rows collected")
	}

	if err := os.MkdirAll(filepath.Dir(outPath), 0o755); err != nil {
		return err
	}

	f, err := os.Create(outPath)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	zw := zip.NewWriter(f)
	defer func() { _ = zw.Close() }()

	n := len(c.y)
	if err := writeNPYToZip(zw, "x_seq.npy", "<f4", []int{n, c.seqLen, c.seqFeatures}, c.xSeq); err != nil {
		return err
	}
	if err := writeNPYToZip(zw, "x_global.npy", "<f4", []int{n, c.globalFeatures}, c.xGlobal); err != nil {
		return err
	}
	if err := writeNPYToZip(zw, "x_seq_mask.npy", "|u1", []int{n, c.seqLen}, c.xSeqMask); err != nil {
		return err
	}
	if err := writeNPYToZip(zw, "y.npy", "<i8", []int{n}, c.y); err != nil {
		return err
	}

	meta := map[string]any{
		"schema_version":       tensorSchemaVersion,
		"num_samples":          n,
		"x_seq_shape":          []int{n, c.seqLen, c.seqFeatures},
		"x_global_shape":       []int{n, c.globalFeatures},
		"x_seq_mask_shape":     []int{n, c.seqLen},
		"y_shape":              []int{n},
		"seq_feature_names":    c.seqFeatureNames,
		"global_feature_names": c.globalFeatureNames,
		"map_vocab":            sortedVocab(c.mapVocab),
		"weapon_vocab":         sortedVocab(c.weaponVocab),
	}
	if err := writeJSONToZip(zw, "metadata.json", meta); err != nil {
		return err
	}
	if err := writeJSONToZip(zw, "map_vocab.json", sortedVocab(c.mapVocab)); err != nil {
		return err
	}
	if err := writeJSONToZip(zw, "weapon_vocab.json", sortedVocab(c.weaponVocab)); err != nil {
		return err
	}

	if err := zw.Close(); err != nil {
		return err
	}
	return nil
}

func writeNPYFloat32(path string, shape []int, data []float32) error {
	return writeNPY(path, "<f4", shape, data)
}

func writeNPYUInt8(path string, shape []int, data []uint8) error {
	return writeNPY(path, "|u1", shape, data)
}

func writeNPYInt64(path string, shape []int, data []int64) error {
	return writeNPY(path, "<i8", shape, data)
}

func writeNPY(path, descr string, shape []int, data any) error {
	b, err := buildNPYBytes(descr, shape, data)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

func writeNPYToZip(zw *zip.Writer, name, descr string, shape []int, data any) error {
	b, err := buildNPYBytes(descr, shape, data)
	if err != nil {
		return err
	}
	w, err := zw.Create(name)
	if err != nil {
		return err
	}
	if _, err := w.Write(b); err != nil {
		return err
	}
	return nil
}

func buildNPYBytes(descr string, shape []int, data any) ([]byte, error) {
	header := buildNPYHeader(descr, shape)
	if len(header) > int(^uint16(0)) {
		return nil, fmt.Errorf("NPY header too large for v1.0 format: %d bytes", len(header))
	}

	buf := &bytes.Buffer{}
	if _, err := buf.Write([]byte{'\x93', 'N', 'U', 'M', 'P', 'Y'}); err != nil {
		return nil, err
	}
	if _, err := buf.Write([]byte{1, 0}); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, uint16(len(header))); err != nil {
		return nil, err
	}
	if _, err := buf.Write(header); err != nil {
		return nil, err
	}
	if err := binary.Write(buf, binary.LittleEndian, data); err != nil {
		return nil, err
	}

	return buf.Bytes(), nil
}

func buildNPYHeader(descr string, shape []int) []byte {
	dict := fmt.Sprintf("{'descr': '%s', 'fortran_order': False, 'shape': %s, }", descr, formatShapeTuple(shape))
	pad := (16 - ((10 + len(dict) + 1) % 16)) % 16
	return []byte(dict + strings.Repeat(" ", pad) + "\n")
}

func formatShapeTuple(shape []int) string {
	if len(shape) == 0 {
		return "()"
	}
	if len(shape) == 1 {
		return fmt.Sprintf("(%d,)", shape[0])
	}

	parts := make([]string, len(shape))
	for i, dim := range shape {
		parts[i] = fmt.Sprintf("%d", dim)
	}
	return "(" + strings.Join(parts, ", ") + ")"
}

func writeJSON(path string, value any) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer func() { _ = f.Close() }()

	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	return enc.Encode(value)
}

func writeJSONToZip(zw *zip.Writer, name string, value any) error {
	w, err := zw.Create(name)
	if err != nil {
		return err
	}
	enc := json.NewEncoder(w)
	enc.SetIndent("", "  ")
	return enc.Encode(value)
}

func sortedVocab(v map[string]int) map[string]int {
	if len(v) == 0 {
		return map[string]int{"<unknown>": 0}
	}
	keys := make([]string, 0, len(v))
	for k := range v {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make(map[string]int, len(v))
	for _, k := range keys {
		out[k] = v[k]
	}
	return out
}

func buildTensorSample(rec pipelineRecord, seqLen int, mapID, weaponID int, avgRankNumeric float64, hasAvgRank bool) (tensorSample, bool) {
	n := minInt(
		len(rec.SeqAngularOffsetDeg),
		len(rec.SeqSpottedFlag),
		len(rec.SeqKillerSpeed),
		len(rec.SeqKillerScopedFlag),
		len(rec.SeqVictimAudibleAnyFlag),
	)
	if n <= 0 {
		return tensorSample{}, false
	}

	xSeq := make([][]float32, seqLen)
	xMask := make([]uint8, seqLen)
	for i := 0; i < seqLen; i++ {
		xSeq[i] = make([]float32, 5)
	}

	pad := 0
	start := 0
	if n >= seqLen {
		start = n - seqLen
	} else {
		pad = seqLen - n
	}

	used := minInt(n, seqLen)
	for i := 0; i < used; i++ {
		src := start + i
		dst := pad + i
		xSeq[dst][0] = float32(rec.SeqAngularOffsetDeg[src])
		xSeq[dst][1] = float32(rec.SeqSpottedFlag[src])
		xSeq[dst][2] = float32(rec.SeqKillerSpeed[src])
		xSeq[dst][3] = float32(rec.SeqKillerScopedFlag[src])
		xSeq[dst][4] = float32(rec.SeqVictimAudibleAnyFlag[src])
		xMask[dst] = 1
	}

	audibleEventCount := rec.VictimAudibleFootsteps + rec.VictimAudibleWeaponFire + rec.VictimAudibleJumps
	hiddenTrackingExists := rec.MinHiddenAngularOffset >= 0
	hiddenAngleRange := float64(0)
	if hiddenTrackingExists {
		hiddenAngleRange = rec.MaxHiddenAngularOffset - rec.MinHiddenAngularOffset
	}
	hiddenTrackingConvergenceStrength := float64(0)
	if rec.HiddenTrackingSlope < 0 {
		hiddenTrackingConvergenceStrength = -rec.HiddenTrackingSlope
	}
	audibleRecentFlag := rec.TicksSinceLastVictimSound >= 0 && rec.TicksSinceLastVictimSound <= 64
	wallbangFlag := rec.PenetratedObjects > 0
	closeRangeFlag := rec.KillDistance <= 500
	longRangeFlag := rec.KillDistance >= 2000
	spottedLowFlag := rec.PctVictimSpotted <= 0.2
	suspiciousAlignmentFlag := hiddenTrackingExists && rec.MinHiddenAngularOffset <= 3.0 && spottedLowFlag

	xGlobal := []float32{
		float32(rec.KillDistance),
		float32(clamp01(rec.PctVictimSpotted)),
		float32(avgRankNumeric),
		boolToFloat32(hasAvgRank),
		float32(mapID),
		float32(weaponID),
		float32(rec.AngularOffsetAtKill),
		float32(rec.KillerAvgSpeed),
		float32(rec.PenetratedObjects),
		boolToFloat32(rec.IsHeadshot),
		boolToFloat32(rec.ThroughSmoke),
		boolToFloat32(rec.KillerScopedAtKill),
		boolToFloat32(hiddenTrackingExists),
		float32(hiddenAngleRange),
		float32(hiddenTrackingConvergenceStrength),
		float32(audibleEventCount),
		boolToFloat32(audibleRecentFlag),
		boolToFloat32(wallbangFlag),
		boolToFloat32(closeRangeFlag),
		boolToFloat32(longRangeFlag),
		boolToFloat32(spottedLowFlag),
		boolToFloat32(suspiciousAlignmentFlag),
	}

	sampleID := fmt.Sprintf("v1:%s:%d:%d:%d", rec.DemoName, rec.Tick, rec.KillerSteamID64, rec.VictimSteamID64)

	return tensorSample{
		SchemaVersion:   tensorSchemaVersion,
		SampleID:        sampleID,
		DemoName:        rec.DemoName,
		DemoPath:        rec.DemoPath,
		KillTick:        rec.Tick,
		Round:           rec.Round,
		KillerSteamID64: rec.KillerSteamID64,
		VictimSteamID64: rec.VictimSteamID64,
		TargetLabel:     rec.Label,
		TargetLabelName: rec.LabelName,
		XSeq:            xSeq,
		XSeqMask:        xMask,
		XGlobal:         xGlobal,
		SeqFeatureNames: []string{
			"ang_offset_deg",
			"spotted_flag",
			"killer_speed",
			"killer_scoped_flag",
			"victim_audible_any_flag",
		},
		GlobalFeatureNames: []string{
			"kill_distance",
			"pct_victim_spotted",
			"avg_rank_numeric",
			"avg_rank_available",
			"map_id",
			"weapon_id",
			"angular_offset_at_kill_deg",
			"killer_avg_speed_units_per_tick",
			"penetrated_objects",
			"is_headshot",
			"through_smoke",
			"killer_scoped_at_kill",
			"hidden_tracking_exists",
			"hidden_angle_range_deg",
			"hidden_tracking_convergence_strength",
			"audible_event_count",
			"audible_recent_flag",
			"wallbang_flag",
			"close_range_flag",
			"long_range_flag",
			"spotted_low_flag",
			"suspicious_alignment_flag",
		},
	}, true
}

func minInt(v int, rest ...int) int {
	m := v
	for _, x := range rest {
		if x < m {
			m = x
		}
	}
	return m
}

func clamp01(v float64) float64 {
	return math.Max(0, math.Min(1, v))
}

func boolToFloat32(v bool) float32 {
	if v {
		return 1
	}
	return 0
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
