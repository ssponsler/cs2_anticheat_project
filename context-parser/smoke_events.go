package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"

	demoinfocs "github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs"
	common "github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs/common"
	events "github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs/events"
)

// go run ./context-parser/smoke_events.go -demo {.dem}

// ML format - JSONL/CSV
// go run ./context-parser/smoke_events.go -demo {.dem} -out-format both -jsonl-out ./smoke_kill_events.jsonl -csv-out ./smoke_kill_events.csv
func main() {
	args := parseArgs()

	f, err := os.Open(args.demoPath)
	checkError(err)
	defer f.Close()

	p := demoinfocs.NewParser(f)
	defer p.Close()

	jsonlWriter, closeJSONL := buildJSONLWriter(args.outFormat, args.jsonlOut)
	defer closeJSONL()

	csvWriter, closeCSV := buildCSVWriter(args.outFormat, args.csvOut)
	defer closeCSV()

	smokeKillCount := 0

	p.RegisterEventHandler(func(e events.Kill) {
		if !e.ThroughSmoke {
			return
		}

		record := smokeKillRecord{
			Tick:              p.GameState().IngameTick(),
			Round:             p.GameState().TotalRoundsPlayed() + 1,
			KillerName:        playerName(e.Killer),
			KillerSteamID64:   playerSteamID64(e.Killer),
			VictimName:        playerName(e.Victim),
			VictimSteamID64:   playerSteamID64(e.Victim),
			Weapon:            fmt.Sprint(e.Weapon),
			PenetratedObjects: e.PenetratedObjects,
			Distance:          e.Distance,
			IsHeadshot:        e.IsHeadshot,
			ThroughSmoke:      e.ThroughSmoke,
		}

		if jsonlWriter != nil {
			checkError(jsonlWriter.Write(record))
		}

		if csvWriter != nil {
			checkError(csvWriter.Write(record))
		}

		smokeKillCount++
		fmt.Printf(
			"[Tick %d | Round %d] %s killed %s | weapon=%s | penetrations=%d | distance=%.1f | hs=%t | smoke=%t\n",
			record.Tick,
			record.Round,
			formatPlayer(e.Killer),
			formatPlayer(e.Victim),
			e.Weapon,
			record.PenetratedObjects,
			record.Distance,
			record.IsHeadshot,
			record.ThroughSmoke,
		)
	})

	err = p.ParseToEnd()
	checkError(err)

	fmt.Printf("\nTotal through-smoke kills: %d\n", smokeKillCount)
	if jsonlWriter != nil {
		fmt.Printf("JSONL written to: %s\n", args.jsonlOut)
	}
	if csvWriter != nil {
		fmt.Printf("CSV written to: %s\n", args.csvOut)
	}
}

type cliArgs struct {
	demoPath  string
	outFormat string
	jsonlOut  string
	csvOut    string
}

func parseArgs() cliArgs {
	demoPath := flag.String("demo", "", "Demo file path")
	outFormat := flag.String("out-format", "jsonl", "Output format: jsonl|csv|both")
	jsonlOut := flag.String("jsonl-out", "smoke_kill_events.jsonl", "Path to JSONL output file")
	csvOut := flag.String("csv-out", "smoke_kill_events.csv", "Path to CSV output file")
	flag.Parse()

	if *demoPath == "" {
		panic("missing required -demo argument")
	}

	if *outFormat != "jsonl" && *outFormat != "csv" && *outFormat != "both" {
		panic("invalid -out-format, expected one of: jsonl, csv, both")
	}

	return cliArgs{
		demoPath:  *demoPath,
		outFormat: *outFormat,
		jsonlOut:  *jsonlOut,
		csvOut:    *csvOut,
	}
}

type smokeKillRecord struct {
	Tick              int     `json:"tick"`
	Round             int     `json:"round"`
	KillerName        string  `json:"killer_name"`
	KillerSteamID64   uint64  `json:"killer_steamid64"`
	VictimName        string  `json:"victim_name"`
	VictimSteamID64   uint64  `json:"victim_steamid64"`
	Weapon            string  `json:"weapon"`
	PenetratedObjects int     `json:"penetrated_objects"`
	Distance          float32 `json:"distance"`
	IsHeadshot        bool    `json:"is_headshot"`
	ThroughSmoke      bool    `json:"through_smoke"`
}

type jsonlWriter struct {
	file    *os.File
	encoder *json.Encoder
}

func buildJSONLWriter(outFormat, path string) (*jsonlWriter, func()) {
	if outFormat != "jsonl" && outFormat != "both" {
		return nil, func() {}
	}

	f, err := os.Create(path)
	checkError(err)

	return &jsonlWriter{
			file:    f,
			encoder: json.NewEncoder(f),
		}, func() {
			checkError(f.Close())
		}
}

func (w *jsonlWriter) Write(record smokeKillRecord) error {
	return w.encoder.Encode(record)
}

type csvWriter struct {
	file   *os.File
	writer *csv.Writer
}

func buildCSVWriter(outFormat, path string) (*csvWriter, func()) {
	if outFormat != "csv" && outFormat != "both" {
		return nil, func() {}
	}

	f, err := os.Create(path)
	checkError(err)

	w := csv.NewWriter(f)
	err = w.Write([]string{
		"tick",
		"round",
		"killer_name",
		"killer_steamid64",
		"victim_name",
		"victim_steamid64",
		"weapon",
		"penetrated_objects",
		"distance",
		"is_headshot",
		"through_smoke",
	})
	checkError(err)

	return &csvWriter{
			file:   f,
			writer: w,
		}, func() {
			w.Flush()
			checkError(w.Error())
			checkError(f.Close())
		}
}

func (w *csvWriter) Write(record smokeKillRecord) error {
	return w.writer.Write([]string{
		strconv.Itoa(record.Tick),
		strconv.Itoa(record.Round),
		record.KillerName,
		strconv.FormatUint(record.KillerSteamID64, 10),
		record.VictimName,
		strconv.FormatUint(record.VictimSteamID64, 10),
		record.Weapon,
		strconv.Itoa(record.PenetratedObjects),
		fmt.Sprintf("%.6f", record.Distance),
		strconv.FormatBool(record.IsHeadshot),
		strconv.FormatBool(record.ThroughSmoke),
	})
}

func playerName(p *common.Player) string {
	if p == nil {
		return ""
	}

	return p.Name
}

func playerSteamID64(p *common.Player) uint64 {
	if p == nil {
		return 0
	}

	return p.SteamID64
}

func formatPlayer(p *common.Player) string {
	if p == nil {
		return "?"
	}

	switch p.Team {
	case common.TeamTerrorists:
		return "[T]" + p.Name
	case common.TeamCounterTerrorists:
		return "[CT]" + p.Name
	default:
		return p.Name
	}
}

func checkError(err error) {
	if err != nil {
		panic(err)
	}
}
