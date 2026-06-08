// Pre-kill context window extractor.
//
// For every kill in a demo, captures a window of per-tick state
// before the kill and computes features for cheat detection:
//
//   - Crosshair angular offset to victim while victim is invisible (hidden tracking)
//   - Linear convergence slope of crosshair toward victim while hidden
//   - Fraction of window where the victim was spotted
//   - Audible sound cues (footsteps, weapon fire, jumps) from victim to killer
//   - Ticks since the last audible victim sound
//   - Killer average movement speed and scoped state at kill
//
// Usage:
//
//	go run ./context-parser/prekill_cw.go -demo <path.dem>
//	go run ./context-parser/prekill_cw.go -demo <path.dem> -out-format both \
//	  -jsonl-out ./prekill.jsonl -csv-out ./prekill.csv
package main

import (
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"math"
	"os"
	"strconv"

	"github.com/golang/geo/r3"
	demoinfocs "github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs"
	events "github.com/markus-wa/demoinfocs-golang/v5/pkg/demoinfocs/events"
)

const (
	windowSeconds  = 3.0
	maxFrameBuffer = 512

	// placeholder, need to determine audibility radius for each cue type through experimentation
	footstepAuditRadius   = 700.0 // running; walking footsteps are client-silent
	weaponFireAuditRadius = 1500.0
	jumpAuditRadius       = 400.0
	reloadAuditRadius     = 400.0
)

// ---- snapshot types

// playerSnapshot holds a single player state at one tick.
type playerSnapshot struct {
	Pos       r3.Vector
	Yaw       float32 // degrees 0-360, from ViewDirectionX()
	Pitch     float32 // degrees raw, from ViewDirectionY() (270-90)
	IsWalking bool
	IsScoped  bool
	Speed     float64         // Euclidean distance from previous tick position
	Spotted   map[uint64]bool // SteamID64s this player has spotted this tick
}

// tickState holds all active player snapshots at one tick.
type tickState struct {
	Tick    int
	Players map[uint64]playerSnapshot // keyed by SteamID64
}

// soundCueEntry records a single audible event emitted by a player.
type soundCueEntry struct {
	Tick      int
	SourceID  uint64
	CueType   string // "footstep" | "weapon_fire" | "jump" | "weapon_reload"
	Pos       r3.Vector
	IsWalking bool // only meaningful for "footstep"
}

// ---- output record ---------------------------------------------------------

// preKillRecord is one training row per kill.
type preKillRecord struct {
	// Kill metadata
	Tick              int     `json:"tick"`
	Round             int     `json:"round"`
	KillerName        string  `json:"killer_name"`
	KillerSteamID64   uint64  `json:"killer_steamid64"`
	VictimName        string  `json:"victim_name"`
	VictimSteamID64   uint64  `json:"victim_steamid64"`
	Weapon            string  `json:"weapon"`
	IsHeadshot        bool    `json:"is_headshot"`
	ThroughSmoke      bool    `json:"through_smoke"`
	PenetratedObjects int     `json:"penetrated_objects"`
	KillDistance      float32 `json:"kill_distance"`

	// window summary
	WindowTicks int `json:"window_ticks"`
	// fraction [0,1] of window ticks where killer had spotted victim.
	PctVictimSpotted float64 `json:"pct_victim_spotted"`
	// crosshair angular error (degrees) at the moment of the kill.
	AngularOffsetAtKill float64 `json:"angular_offset_at_kill_deg"`
	// min crosshair angular error (degrees) during ticks where victim was NOT spotted.
	// -1 when victim was spotted for the entire window.
	MinHiddenAngularOffset float64 `json:"min_hidden_angular_offset_deg"`
	// maximum crosshair angular error during hidden ticks.
	MaxHiddenAngularOffset float64 `json:"max_hidden_angular_offset_deg"`
	// OLS slope of angular offset vs tick index while victim was hidden.
	// negative = crosshair converging on victim through wall.
	HiddenTrackingSlope float64 `json:"hidden_tracking_slope_deg_per_tick"`

	// sound cues audible to killer during window
	VictimAudibleFootsteps  int `json:"victim_audible_footsteps"`
	VictimAudibleWeaponFire int `json:"victim_audible_weapon_fire"`
	VictimAudibleJumps      int `json:"victim_audible_jumps"`
	// ticks between the last audible victim event and the kill. -1 if none.
	TicksSinceLastVictimSound int `json:"ticks_since_last_victim_sound"`

	// killer behavior
	KillerAvgSpeed     float64 `json:"killer_avg_speed_units_per_tick"`
	KillerScopedAtKill bool    `json:"killer_scoped_at_kill"`

	// Optional per-tick sequence channels (window order: oldest -> kill tick).
	SeqAngularOffsetDeg     []float32 `json:"seq_ang_offset_deg,omitempty"`
	SeqSpottedFlag          []uint8   `json:"seq_spotted_flag,omitempty"`
	SeqKillerSpeed          []float32 `json:"seq_killer_speed,omitempty"`
	SeqKillerScopedFlag     []uint8   `json:"seq_killer_scoped_flag,omitempty"`
	SeqVictimAudibleAnyFlag []uint8   `json:"seq_victim_audible_any_flag,omitempty"`
}

type cliArgs struct {
	demoPath   string
	outFormat  string
	jsonlOut   string
	csvOut     string
	includeSeq bool
}

func parseArgs() cliArgs {
	demoPath := flag.String("demo", "", "Demo file path")
	outFormat := flag.String("out-format", "jsonl", "Output format: jsonl|csv|both")
	jsonlOut := flag.String("jsonl-out", "prekill_cw.jsonl", "Path to JSONL output file")
	csvOut := flag.String("csv-out", "prekill_cw.csv", "Path to CSV output file")
	includeSeq := flag.Bool("include-seq", false, "Include per-tick sequence channels in JSONL output")
	flag.Parse()

	if *demoPath == "" {
		panic("missing required -demo argument")
	}
	if *outFormat != "jsonl" && *outFormat != "csv" && *outFormat != "both" {
		panic("invalid -out-format; expected one of: jsonl, csv, both")
	}
	return cliArgs{*demoPath, *outFormat, *jsonlOut, *csvOut, *includeSeq}
}

func main() {
	args := parseArgs()

	f, err := os.Open(args.demoPath)
	checkError(err)
	defer f.Close()

	p := demoinfocs.NewParser(f)
	defer p.Close()

	jw, closeJSONL := buildJSONLWriter(args.outFormat, args.jsonlOut)
	defer closeJSONL()

	cw, closeCSV := buildCSVWriter(args.outFormat, args.csvOut)
	defer closeCSV()

	var frameBuffer []tickState
	var soundCues []soundCueEntry
	prevPositions := map[uint64]r3.Vector{}
	totalKills := 0

	// --- per-tick state snapshot ---
	p.RegisterEventHandler(func(e events.FrameDone) {
		tick := p.GameState().IngameTick()
		playing := p.GameState().Participants().Playing()

		state := tickState{
			Tick:    tick,
			Players: make(map[uint64]playerSnapshot, len(playing)),
		}

		for _, pl := range playing {
			if pl == nil || pl.SteamID64 == 0 {
				continue
			}
			pos := pl.Position()

			speed := 0.0
			if prev, ok := prevPositions[pl.SteamID64]; ok {
				speed = pos.Sub(prev).Norm()
			}
			prevPositions[pl.SteamID64] = pos

			// Build spotted set: which other players has this player spotted.
			spotted := make(map[uint64]bool, len(playing))
			for _, other := range playing {
				if other != nil && other.SteamID64 != 0 && other.SteamID64 != pl.SteamID64 {
					if pl.HasSpotted(other) {
						spotted[other.SteamID64] = true
					}
				}
			}

			state.Players[pl.SteamID64] = playerSnapshot{
				Pos:       pos,
				Yaw:       pl.ViewDirectionX(),
				Pitch:     pl.ViewDirectionY(),
				IsWalking: pl.IsWalking(),
				IsScoped:  pl.IsScoped(),
				Speed:     speed,
				Spotted:   spotted,
			}
		}

		frameBuffer = append(frameBuffer, state)
		if len(frameBuffer) > maxFrameBuffer {
			frameBuffer = frameBuffer[len(frameBuffer)-maxFrameBuffer:]
		}
	})

	// --- sound cue capture ---
	p.RegisterEventHandler(func(e events.Footstep) {
		if e.Player == nil {
			return
		}
		soundCues = append(soundCues, soundCueEntry{
			Tick:      p.GameState().IngameTick(),
			SourceID:  e.Player.SteamID64,
			CueType:   "footstep",
			Pos:       e.Player.Position(),
			IsWalking: e.Player.IsWalking(),
		})
	})

	p.RegisterEventHandler(func(e events.WeaponFire) {
		if e.Shooter == nil {
			return
		}
		soundCues = append(soundCues, soundCueEntry{
			Tick:     p.GameState().IngameTick(),
			SourceID: e.Shooter.SteamID64,
			CueType:  "weapon_fire",
			Pos:      e.Shooter.Position(),
		})
	})

	p.RegisterEventHandler(func(e events.PlayerJump) {
		if e.Player == nil {
			return
		}
		soundCues = append(soundCues, soundCueEntry{
			Tick:     p.GameState().IngameTick(),
			SourceID: e.Player.SteamID64,
			CueType:  "jump",
			Pos:      e.Player.Position(),
		})
	})

	p.RegisterEventHandler(func(e events.WeaponReload) {
		if e.Player == nil {
			return
		}
		soundCues = append(soundCues, soundCueEntry{
			Tick:     p.GameState().IngameTick(),
			SourceID: e.Player.SteamID64,
			CueType:  "weapon_reload",
			Pos:      e.Player.Position(),
		})
	})

	// --- kill handler: build window and emit record ---
	p.RegisterEventHandler(func(e events.Kill) {
		if e.Killer == nil || e.Victim == nil {
			return
		}

		killTick := p.GameState().IngameTick()
		tickRate := p.TickRate()
		if tickRate <= 0 {
			tickRate = 64.0
		}

		windowStartTick := killTick - int(windowSeconds*tickRate)

		// Collect frames inside the window.
		windowFrames := make([]tickState, 0, int(windowSeconds*tickRate))
		for _, frame := range frameBuffer {
			if frame.Tick >= windowStartTick && frame.Tick <= killTick {
				windowFrames = append(windowFrames, frame)
			}
		}

		// Collect sound cues inside the window.
		windowCues := make([]soundCueEntry, 0)
		for _, cue := range soundCues {
			if cue.Tick >= windowStartTick && cue.Tick <= killTick {
				windowCues = append(windowCues, cue)
			}
		}

		round := p.GameState().TotalRoundsPlayed() + 1
		killerScopedAtKill := e.Killer.IsScoped()

		rec := buildPreKillRecord(e, killTick, round, windowFrames, windowCues, killerScopedAtKill, args.includeSeq)

		fmt.Printf(
			"[Tick %d | Round %d] %s -> %s | weapon=%s | spotted=%.0f%% | min_hidden_angle=%.1f | sound_cues=%d | hidden_slope=%.4f\n",
			rec.Tick, rec.Round,
			rec.KillerName, rec.VictimName,
			rec.Weapon,
			rec.PctVictimSpotted*100,
			rec.MinHiddenAngularOffset,
			rec.VictimAudibleFootsteps+rec.VictimAudibleWeaponFire+rec.VictimAudibleJumps,
			rec.HiddenTrackingSlope,
		)

		if jw != nil {
			checkError(jw.Write(rec))
		}
		if cw != nil {
			checkError(cw.Write(rec))
		}
		totalKills++

		// Prune sound cues older than 10s to bound memory.
		pruneBeforeTick := killTick - int(10.0*tickRate)
		pruned := soundCues[:0]
		for _, cue := range soundCues {
			if cue.Tick >= pruneBeforeTick {
				pruned = append(pruned, cue)
			}
		}
		soundCues = pruned
	})

	checkError(p.ParseToEnd())

	fmt.Printf("\nTotal kills processed: %d\n", totalKills)
	if jw != nil {
		fmt.Printf("JSONL written to: %s\n", args.jsonlOut)
	}
	if cw != nil {
		fmt.Printf("CSV written to: %s\n", args.csvOut)
	}
}

// ---- window feature computation

func buildPreKillRecord(
	e events.Kill,
	killTick, round int,
	windowFrames []tickState,
	soundCues []soundCueEntry,
	killerScopedAtKill bool,
	includeSeq bool,
) preKillRecord {
	killerID := e.Killer.SteamID64
	victimID := e.Victim.SteamID64

	rec := preKillRecord{
		Tick:                      killTick,
		Round:                     round,
		KillerName:                e.Killer.Name,
		KillerSteamID64:           killerID,
		VictimName:                e.Victim.Name,
		VictimSteamID64:           victimID,
		Weapon:                    fmt.Sprint(e.Weapon),
		IsHeadshot:                e.IsHeadshot,
		ThroughSmoke:              e.ThroughSmoke,
		PenetratedObjects:         e.PenetratedObjects,
		KillDistance:              e.Distance,
		WindowTicks:               len(windowFrames),
		MinHiddenAngularOffset:    -1,
		MaxHiddenAngularOffset:    -1,
		TicksSinceLastVictimSound: -1,
		KillerScopedAtKill:        killerScopedAtKill,
	}

	spottedCount := 0
	totalSpeed := 0.0
	speedCount := 0
	hiddenIdxs := []float64{}
	hiddenAngles := []float64{}
	audibleByTick := map[int]bool{}

	if includeSeq {
		for _, cue := range soundCues {
			if cue.SourceID != victimID {
				continue
			}
			if cue.CueType == "footstep" && cue.IsWalking {
				continue
			}

			killerPos := killerPosAtTick(windowFrames, killerID, cue.Tick)

			var radius float64
			switch cue.CueType {
			case "footstep":
				radius = footstepAuditRadius
			case "weapon_fire":
				radius = weaponFireAuditRadius
			case "jump":
				radius = jumpAuditRadius
			case "weapon_reload":
				radius = reloadAuditRadius
			default:
				radius = 500.0
			}

			if isAudible(killerPos, cue.Pos, radius) {
				audibleByTick[cue.Tick] = true
			}
		}
	}

	for i, frame := range windowFrames {
		killerSnap, kOk := frame.Players[killerID]
		victimSnap, vOk := frame.Players[victimID]
		if !kOk || !vOk {
			continue
		}

		isSpotted := killerSnap.Spotted[victimID]
		if isSpotted {
			spottedCount++
		}

		angOffset := angularOffsetDeg(killerSnap.Pos, killerSnap.Yaw, killerSnap.Pitch, victimSnap.Pos)

		if !isSpotted {
			if rec.MinHiddenAngularOffset < 0 || angOffset < rec.MinHiddenAngularOffset {
				rec.MinHiddenAngularOffset = angOffset
			}
			if angOffset > rec.MaxHiddenAngularOffset {
				rec.MaxHiddenAngularOffset = angOffset
			}
			hiddenIdxs = append(hiddenIdxs, float64(i))
			hiddenAngles = append(hiddenAngles, angOffset)
		}

		// The last frame in the window is closest to the kill tick.
		if i == len(windowFrames)-1 {
			rec.AngularOffsetAtKill = angOffset
		}

		if killerSnap.Speed > 0 {
			totalSpeed += killerSnap.Speed
			speedCount++
		}

		if includeSeq {
			rec.SeqAngularOffsetDeg = append(rec.SeqAngularOffsetDeg, float32(angOffset))
			rec.SeqSpottedFlag = append(rec.SeqSpottedFlag, boolToUint8(isSpotted))
			rec.SeqKillerSpeed = append(rec.SeqKillerSpeed, float32(killerSnap.Speed))
			rec.SeqKillerScopedFlag = append(rec.SeqKillerScopedFlag, boolToUint8(killerSnap.IsScoped))
			rec.SeqVictimAudibleAnyFlag = append(rec.SeqVictimAudibleAnyFlag, boolToUint8(audibleByTick[frame.Tick]))
		}
	}

	if len(windowFrames) > 0 {
		rec.PctVictimSpotted = float64(spottedCount) / float64(len(windowFrames))
	}
	if speedCount > 0 {
		rec.KillerAvgSpeed = totalSpeed / float64(speedCount)
	}
	rec.HiddenTrackingSlope = linearSlope(hiddenIdxs, hiddenAngles)

	// Sound cue analysis.
	lastSoundTick := -1
	for _, cue := range soundCues {
		if cue.SourceID != victimID {
			continue
		}
		// Walking footsteps are silent in CS2.
		if cue.CueType == "footstep" && cue.IsWalking {
			continue
		}

		killerPos := killerPosAtTick(windowFrames, killerID, cue.Tick)

		var radius float64
		switch cue.CueType {
		case "footstep":
			radius = footstepAuditRadius
		case "weapon_fire":
			radius = weaponFireAuditRadius
		case "jump":
			radius = jumpAuditRadius
		case "weapon_reload":
			radius = reloadAuditRadius
		default:
			radius = 500.0
		}

		if isAudible(killerPos, cue.Pos, radius) {
			switch cue.CueType {
			case "footstep":
				rec.VictimAudibleFootsteps++
			case "weapon_fire":
				rec.VictimAudibleWeaponFire++
			case "jump":
				rec.VictimAudibleJumps++
			}
			if cue.Tick > lastSoundTick {
				lastSoundTick = cue.Tick
			}
		}
	}

	if lastSoundTick >= 0 {
		rec.TicksSinceLastVictimSound = killTick - lastSoundTick
	}

	return rec
}

// angularOffsetDeg returns the angle in degrees between the killer's forward
// view vector and the direction toward the victim's position.
func angularOffsetDeg(killerPos r3.Vector, yaw, pitch float32, victimPos r3.Vector) float64 {
	// Normalise pitch to [-90, 90]: ViewDirectionY returns 270..360 for upward look.
	p := float64(pitch)
	if p > 180 {
		p -= 360
	}
	yawRad := float64(yaw) * math.Pi / 180.0
	pitchRad := p * math.Pi / 180.0

	// Killer forward unit vector in world space.
	fx := math.Cos(pitchRad) * math.Cos(yawRad)
	fy := math.Cos(pitchRad) * math.Sin(yawRad)
	fz := -math.Sin(pitchRad) // Z-up; looking up → positive pitch → negative fz

	// Unit vector from killer to victim.
	delta := victimPos.Sub(killerPos)
	dist := delta.Norm()
	if dist < 1e-6 {
		return 0
	}
	dx := delta.X / dist
	dy := delta.Y / dist
	dz := delta.Z / dist

	dot := math.Max(-1, math.Min(1, fx*dx+fy*dy+fz*dz))
	return math.Acos(dot) * 180.0 / math.Pi
}

// isAudible returns true if the source position is within the audibility radius.
func isAudible(killerPos, sourcePos r3.Vector, radius float64) bool {
	return killerPos.Sub(sourcePos).Norm() <= radius
}

// killerPosAtTick returns the killer's position at the tick closest to target.
func killerPosAtTick(frames []tickState, killerID uint64, targetTick int) r3.Vector {
	best := r3.Vector{}
	bestDist := math.MaxInt32
	for _, f := range frames {
		d := f.Tick - targetTick
		if d < 0 {
			d = -d
		}
		if d < bestDist {
			bestDist = d
			if snap, ok := f.Players[killerID]; ok {
				best = snap.Pos
			}
		}
	}
	return best
}

// linearSlope computes the OLS regression slope of ys ~ xs.
// Returns 0 if fewer than 2 data points.
func linearSlope(xs, ys []float64) float64 {
	n := float64(len(xs))
	if n < 2 {
		return 0
	}
	sumX, sumY, sumXY, sumXX := 0.0, 0.0, 0.0, 0.0
	for i := range xs {
		sumX += xs[i]
		sumY += ys[i]
		sumXY += xs[i] * ys[i]
		sumXX += xs[i] * xs[i]
	}
	denom := n*sumXX - sumX*sumX
	if denom == 0 {
		return 0
	}
	return (n*sumXY - sumX*sumY) / denom
}

// ---- JSONL writer ----------------------------------------------------------

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
	return &jsonlWriter{f, json.NewEncoder(f)}, func() { checkError(f.Close()) }
}

func (w *jsonlWriter) Write(rec preKillRecord) error {
	return w.encoder.Encode(rec)
}

// ---- CSV writer ------------------------------------------------------------

var csvHeader = []string{
	"tick", "round",
	"killer_name", "killer_steamid64",
	"victim_name", "victim_steamid64",
	"weapon", "is_headshot", "through_smoke", "penetrated_objects", "kill_distance",
	"window_ticks", "pct_victim_spotted",
	"angular_offset_at_kill_deg",
	"min_hidden_angular_offset_deg", "max_hidden_angular_offset_deg",
	"hidden_tracking_slope_deg_per_tick",
	"victim_audible_footsteps", "victim_audible_weapon_fire", "victim_audible_jumps",
	"ticks_since_last_victim_sound",
	"killer_avg_speed_units_per_tick", "killer_scoped_at_kill",
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
	checkError(w.Write(csvHeader))
	return &csvWriter{f, w}, func() {
		w.Flush()
		checkError(w.Error())
		checkError(f.Close())
	}
}

func (w *csvWriter) Write(rec preKillRecord) error {
	return w.writer.Write([]string{
		strconv.Itoa(rec.Tick),
		strconv.Itoa(rec.Round),
		rec.KillerName,
		strconv.FormatUint(rec.KillerSteamID64, 10),
		rec.VictimName,
		strconv.FormatUint(rec.VictimSteamID64, 10),
		rec.Weapon,
		strconv.FormatBool(rec.IsHeadshot),
		strconv.FormatBool(rec.ThroughSmoke),
		strconv.Itoa(rec.PenetratedObjects),
		fmt.Sprintf("%.6f", rec.KillDistance),
		strconv.Itoa(rec.WindowTicks),
		fmt.Sprintf("%.6f", rec.PctVictimSpotted),
		fmt.Sprintf("%.4f", rec.AngularOffsetAtKill),
		fmt.Sprintf("%.4f", rec.MinHiddenAngularOffset),
		fmt.Sprintf("%.4f", rec.MaxHiddenAngularOffset),
		fmt.Sprintf("%.6f", rec.HiddenTrackingSlope),
		strconv.Itoa(rec.VictimAudibleFootsteps),
		strconv.Itoa(rec.VictimAudibleWeaponFire),
		strconv.Itoa(rec.VictimAudibleJumps),
		strconv.Itoa(rec.TicksSinceLastVictimSound),
		fmt.Sprintf("%.4f", rec.KillerAvgSpeed),
		strconv.FormatBool(rec.KillerScopedAtKill),
	})
}

// ---- misc ------------------------------------------------------------------

func checkError(err error) {
	if err != nil {
		panic(err)
	}
}

func boolToUint8(v bool) uint8 {
	if v {
		return 1
	}
	return 0
}
