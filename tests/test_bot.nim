## The scripted baselines must play whole episodes without ever proposing an
## illegal decision — they are both the no-credentials fallback (offline
## certification) and fieldable policies, so this is the completion path.
## The version-space predictor must also actually learn the rule, or it is no
## partner worth beating.

import std/[json, monotimes, options, os, strutils, times, unicode, unittest]
import eleusis/[llm, server, sim]

proc fixture(seed: int, rounds = 24, testEvery = 6): GameConfig =
  result = defaultGameConfig()
  result.seed = seed
  result.rounds = rounds
  result.testEvery = testEvery
  result.sampled = true
  for index in 0 ..< Seats:
    result.players.add(PlayerConfig(name: "P" & $(index + 1)))
    result.tokens.add("t" & $index)

proc checkLegal(sim: Sim, decision: Decision) =
  ## Every decision a baseline emits must be legal BEFORE the sim is asked to
  ## apply it: that is what makes it a safe fallback.
  check decision.hypothesis.runeLen <= MaxHypothesisLen
  check decision.hypothesis.validateUtf8() == -1
  check decision.notes.len == 0
  if sim.phase == phTest:
    check decision.answers.len == sim.config.testStrips
  else:
    if decision.strip.len > 0:
      check decision.strip.len == StripLen
      for letter in decision.strip:
        check letter in Colours
      check indexOfStrip(decision.strip) >= 0

proc playScripted(config: GameConfig, kinds: array[Seats, ScriptKind]): Sim =
  result = initSim(config)
  while not result.done:
    let testing = result.phase == phTest
    for seat in result.pendingSeats():
      let decision = scriptedAction(result, seat, kinds[seat])
      checkLegal(result, decision)
      if testing:
        result.applyAnswers(seat, decision.answers, decision.publish,
          decision.hypothesis, decision.notes, true)
      else:
        result.applyResearch(seat, decision.strip, decision.publish,
          decision.hypothesis, decision.notes, true)

suite "the play clock":
  test "the play deadline is never switched off":
    ## The deadline is what makes the episode settle and score inside 60% of
    ## the platform's timeout. A non-positive value anywhere in the chain must
    ## fall through to the next source, never to "no deadline at all".
    var config = fixture(1)
    check playTimeoutSeconds(config, "600") == 600.0
    check playTimeoutSeconds(config, "  600  ") == 600.0
    ## Env garbage and an absent env fall back to the configured assumption.
    check playTimeoutSeconds(config, "not-a-number") ==
      config.episodeTimeoutSeconds.float
    check playTimeoutSeconds(config, "") == config.episodeTimeoutSeconds.float
    check playTimeoutSeconds(config, "0") == config.episodeTimeoutSeconds.float
    ## And a config whose own timeout is unusable falls through to the
    ## built-in default rather than leaving the episode unbounded.
    config.episodeTimeoutSeconds = 0
    check playTimeoutSeconds(config, "") ==
      defaultGameConfig().episodeTimeoutSeconds.float
    config.episodeTimeoutSeconds = -5
    check playTimeoutSeconds(config, "-1") ==
      defaultGameConfig().episodeTimeoutSeconds.float
    check playTimeoutSeconds(config, "") > 0.0

suite "scripted baselines":
  test "five baseline seats play full, legal episodes":
    for seed in [1, 7, 42, 99, 1234]:
      let config = fixture(seed, rounds = 12, testEvery = 4)
      let started = getMonoTime()
      let sim = playScripted(config,
        [skOpenbook, skHoarder, skOpenbook, skHoarder, skOpenbook])
      let elapsed = (getMonoTime() - started).inMilliseconds
      check sim.done
      check sim.reason == "complete"
      check sim.roundsPlayed == config.rounds
      check sim.testsDone == config.rounds div config.testEvery
      var experiments = 0
      for event in sim.events:
        if event.kind == evExperiment:
          inc experiments
          check event.strip.len == StripLen
      check experiments > 0
      echo "seed ", seed, ": rule ", sim.ruleText(), ", ", experiments,
        " experiments, board ", sim.board.len, ", ", elapsed, " ms"

  test "the version space really learns: >= 70% on the final test":
    ## Five open-book seats pooling their results should have the rule
    ## cornered by the last test; a coin flip would score 50%.
    var total = 0
    var answered = 0
    for seed in 0 ..< 10:
      let sim = playScripted(fixture(seed, rounds = 12, testEvery = 4),
        [skOpenbook, skOpenbook, skOpenbook, skOpenbook, skOpenbook])
      check sim.testCorrect.len > 0
      for correct in sim.testCorrect[^1]:
        total += correct
        answered += sim.config.testStrips
    let rate = total.float / answered.float
    echo "final-test accuracy over 10 seeds: ", rate
    check rate >= 0.70

  test "hoarders publish nothing; open books publish everything they can":
    let sim = playScripted(fixture(5, rounds = 12, testEvery = 4),
      [skOpenbook, skHoarder, skOpenbook, skHoarder, skOpenbook])
    for seat in [1, 3]:
      check sim.seats[seat].published == 0
      check sim.seats[seat].hoarded > 0
      for fact in sim.board:
        check fact.author != seat
      for fact in sim.seats[seat].log:
        check fact.mode in ["hoard", ""]
    for seat in [0, 2, 4]:
      check sim.seats[seat].hoarded == 0
      check sim.seats[seat].published > 0
      ## Every disclosed result was published; only an exact duplicate of a
      ## strip already on the board fails to earn authorship.
      for fact in sim.seats[seat].log:
        check fact.mode in ["publish", "duplicate", ""]

  test "free riders use the public board without buying experiments":
    let sim = playScripted(fixture(5, rounds = 12, testEvery = 4),
      [skFreerider, skOpenbook, skOpenbook, skOpenbook, skOpenbook])
    check sim.seats[0].experiments == 0
    check sim.seats[0].spend == 0
    check sim.seats[0].answered == 3 * sim.config.testStrips

  test "decideAll plays explicit scripted seats without credentials and fails LLM seats as infra":
    let config = fixture(3, rounds = 8, testEvery = 4)
    let client = newLlmClient(config)
    client.disabled = true
    check client.disabled
    var sim = initSim(config)
    let seats = sim.pendingSeats()
    let decisions = client.decideAll(sim, seats,
      @["be bold", "play smart", "", "", ""],
      @[skNone, skNone, skHoarder, skNone, skOpenbook], PromptSelection())
    check decisions.len == Seats
    ## Explicit scripted seats still produce legal baseline moves.
    for seat in [2, 4]:
      let kind = if seat == 2: skHoarder else: skOpenbook
      check not decisions[seat].failed
      check decisions[seat].strip == scriptedAction(sim, seat, kind).strip
      check decisions[seat].publish == (seat != 2)
    ## Connected LLM seats with prompts fail as infra when credentials are missing.
    for seat in [0, 1]:
      check decisions[seat].failed
      check decisions[seat].failureKind == fkInfra
      check decisions[seat].failureReason == "no LLM credentials configured"
    ## An LLM seat with no prompt delivered fails as player-attributable.
    check decisions[3].failed
    check decisions[3].failureKind == fkPlayer
    check decisions[3].failureReason == "no prompt delivered for seat"

  test "a slot that never delivered a prompt fails instead of silently playing openbook":
    putEnv("ANTHROPIC_API_KEY", "not-a-real-key")
    let config = fixture(17, rounds = 8, testEvery = 4)
    let client = newLlmClient(config)
    check not client.disabled
    var sim = initSim(config)
    let seats = sim.pendingSeats()
    let decisions = client.decideAll(sim, seats,
      @["", "  ", "", "", ""], @[skNone, skNone, skNone, skNone, skNone], PromptSelection())
    check decisions.len == Seats
    for index, seat in seats:
      check decisions[index].failed
      check decisions[index].failureKind == fkPlayer
      check decisions[index].failureReason.len > 0
      check decisions[index].attempts.len == 0
    delEnv("ANTHROPIC_API_KEY")

  test "model replies parse strictly, and legacy shims or prose are rejected":
    check parseDecision(parseJson("""{"experiment": "RBGY"}"""), 6,
      false).strip == "RBGY"
    check parseDecision(parseJson("""{"experiment": ""}"""), 6,
      false).strip == ""
    let full = parseDecision(parseJson(
      """{"experiment": "RRBG", "publish": true, "hypothesis": "ADJACENT R B?",
          "notes": "shortlist: 3"}"""), 6, false)
    check full.strip == "RRBG"
    check full.publish
    check full.hypothesis == "ADJACENT R B?"
    check full.notes == "shortlist: 3"
    ## Optional fields default cleanly when omitted.
    check not parseDecision(parseJson("""{"experiment": "RRBG"}"""), 6,
      false).publish
    ## Legacy shims, aliases, wrong types, and unknown fields are rejected.
    for badResearch in [
      """{"experiment": "rb gy"}""",
      """{"experiment": "R-B-G-Y"}""",
      """{"experiment": "rbgy"}""",
      """{"experiment": "RBGYR"}""",
      """{"experiment": "RBG"}""",
      """{"experiment": "RBGZ"}""",
      """{"experiment": "RRBG", "publish": "yes"}""",
      """{"experiment": "RRBG", "publish": "true"}""",
      """{"experiment": "RRBG", "publish": 1}""",
      """{"experiment": "RRBG", "notes": 123}""",
      """{"experiment": "RRBG", "hypothesis": true}""",
      """{"experiment": "RRBG", "extra": "nope"}""",
      """{"experiment": "RRBG", "answers": ["PASS","FAIL","PASS","FAIL","PASS","FAIL"]}""",
      """{"notes": "no experiment"}""",
      """[1, 2]"""
    ]:
      expect EleusisError:
        discard parseDecision(parseJson(badResearch), 6, false)
    ## Test turns: exactly testStrips entries of PASS/FAIL (or canonical pass/fail).
    let answered = parseDecision(parseJson(
      """{"answers": ["PASS", "fail", "pass", "FAIL", "PASS", "FAIL"], "publish": false}"""), 6, true)
    check answered.answers == @[vPass, vFail, vPass, vFail, vPass, vFail]
    for badTest in [
      """{"answers": ["PASS", "FAIL", "PASS", "FAIL", "PASS"]}""",
      """{"answers": ["PASS", "FAIL", "PASS", "FAIL", "PASS", "maybe"]}""",
      """{"answers": ["P", "F", "PASS", "FAIL", "PASS", "FAIL"]}""",
      """{"answers": [true, false, true, false, true, false]}""",
      """{"answers": ["yes", "no", "PASS", "FAIL", "PASS", "FAIL"]}""",
      """{"experiment": "RBGY"}""",
      """{"answers": ["PASS","FAIL","PASS","FAIL","PASS","FAIL"], "experiment": ""}"""
    ]:
      expect EleusisError:
        discard parseDecision(parseJson(badTest), 6, true)
    ## Rune-safe truncation of everything that lands in the replay.
    var long = ""
    for index in 0 ..< 900:
      long.add("é")
    check cleanText(long, MaxNotesLen).runeLen == MaxNotesLen
    check cleanText(long, MaxHypothesisLen).runeLen == MaxHypothesisLen
    check cleanText(long, MaxNotesLen).validateUtf8() == -1
    ## Strict scripted policy names (no legacy aliases).
    check parseScriptKind("openbook") == skOpenbook
    check parseScriptKind("hoarder") == skHoarder
    check parseScriptKind("freerider") == skFreerider
    check parseScriptKind("") == skNone
    for badScript in ["1", "true", "yes", "open-book", "open", "secretive", "free-rider", "typo"]:
      expect EleusisError:
        discard parseScriptKind(badScript)
    ## JSON extraction: bare object or single ordinary outer markdown code fence only.
    check extractJsonObject("""{"experiment": "RBGY"}"""){"experiment"}.getStr() == "RBGY"
    check extractJsonObject("```\n{\"experiment\": \"RBGY\"}\n```"){"experiment"}.getStr() == "RBGY"
    check extractJsonObject("```json\n{\"experiment\": \"RBGY\"}\n```"){"experiment"}.getStr() == "RBGY"
    check extractJsonObject("```JSON\n{\"experiment\": \"RBGY\"}\n```"){"experiment"}.getStr() == "RBGY"
    for badReply in [
      "prose {\"experiment\": \"RBGY\"} tail",
      "Here is my move:\n```json\n{\"experiment\": \"RBGY\"}\n```",
      "```json\n{\"experiment\": \"RBGY\"}\n```\nDone.",
      "{\"experiment\": \"RBGY\"} {\"experiment\": \"RRRR\"}",
      "```json\n{\"experiment\": \"RBGY\"}\n```\n```json\n{\"experiment\": \"RRRR\"}\n```",
      "```python\n{\"experiment\": \"RBGY\"}\n```",
      "```json\n{\"experiment\": \"RBGY\"}",
      "[1, 2]",
      ""
    ]:
      expect EleusisError:
        discard extractJsonObject(badReply)

  test "prompts carry the seat's own view and nothing hidden":
    var sim = initSim(fixture(7, rounds = 8, testEvery = 4))
    for seat in sim.pendingSeats():
      sim.applyResearch(seat, stripOfIndex(seat * 51), seat == 0,
        "hypothesis " & $seat, "note " & $seat, true)
    for seat in sim.pendingSeats():
      sim.applyResearch(seat, stripOfIndex(seat * 51 + 7), seat == 0,
        "hypothesis " & $seat, "", true)
    let text = sim.userPrompt(0, "operator says hi")
    check "ROUND 3 OF 8" in text
    check "operator says hi" in text
    check "YOUR PENDING RESULT" in text
    check "THE CORKBOARD" in text
    check "SCOREBOARD" in text
    check "note 0" in text
    ## The rule, the other seats' notes and their drawers are never in it.
    check sim.ruleText() notin text
    check "note 1" notin text
    check text.runeLen > 200
    let system = sim.systemPrompt(0, PromptSelection())
    check "ELEUSIS laboratory" in system
    check "reply with ONLY one JSON object" in system
    check sim.ruleText() notin system
