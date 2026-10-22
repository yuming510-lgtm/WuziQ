# V6 execution ledger

Base: `research/abcd-ladder-v5-1` at `d71227c1ee4055e4cb0feee52673b117b20fd3b5`.
Branch: `research/advanced-search-v6`.

- Preflight: baseline suite 220/220 passed in 10.82s.
- Preflight: V5.1 source has 147 game records across 73 full groups and one unpaired observation; 73 full groups are candidates for replay-validated exact reuse, the partial is excluded.
- Preflight: Drive folder created at `https://drive.google.com/drive/folders/190pb6fdnDlcDXkYjMxam-fPBU-flt9l7`; probe upload/download SHA-256 `523eb58475b00605a121e8756d0b9c37f89ba7756eed46fadbb9ed6d661178fe`, empty-directory restore passed, delete passed, folder readback empty.
- Ruling: V5.1 had 93 inferred groups, but V6 will reuse only full empirical pairs. The inference outcomes are carried only as audit metadata and do not enter V6 scores.
- Exact V5.1 reuse audit: 73 paired groups / 146 complete games imported; the unmatched game `v5-1-9x9-a-natural-c-50s-o731110-s0` is excluded. Historical inferred route rows are not outcomes.
- Tactical suite: 12 frozen alternating, nonterminal 9x9 positions; 60 decisions across E1–E5. 43 declared oracle checks were scored, all 43 correct; 17 proof claims independently replayed, 0 invalid; 0 illegal moves or board-restoration failures. E3 did not prove the cross-open-three puzzle within its 2s validation cap; E2 did. The suite does not count “no proof found” as a loss.
- E5 ablation: 18 runs on 3 fixed positions, six variants, depth limit 3 / 2s per move. Same completed-depth evaluations matched; 10/18 completed depth 3, the rest timed out at depth 2. Raw JSON and CSV saved under `research/v6/validation/`.
- Preflight environment: 3 physical/logical CPUs, 9.74 GiB total RAM / about 9.09 GiB available, 31.81 GB free disk. Representative cross-threat position benchmarked B/C/D/E1–E5 at 50ms, 200ms, 1s and 5s; A measured at natural time. Five-second results: B 67,019 nodes/depth 2; C 30,693/depth 4; D 1,359 simulations; E1 1 node/0.003s; E2 73 nodes/0.685s; E3 44 nodes/depth 8; E4 1,189 nodes/depth 2; E5 7,209 nodes/depth 3.
- Preflight forecast: 1,428 maximum paired groups / 2,856 games; current route frontier 360 groups, 1,260 unique groups reachable across still-active routes; 73 groups reused. Scenario wall estimates: optimistic 3.84h (frontier only, 20-ply games and >5s capped at measured 5s), central 70.39h (half reachable groups, V5.1 mean 26.70 plies), conservative 642.96h (all reachable, full 81-ply boards, each search consumes its configured budget). These are workload estimates, not completion claims; update from actual game timings.
- Storage estimate: 134.65 MiB uncompressed at the maximum 2,856 games, using four times the measured V5.1 average game record size plus schedule, report, and progress reserve; available disk is about 31.8 GB.
- Full test suite before the final reserve tuning: 255 passed in 18.17s; post-tuning targeted suite also passed (28 tests). A final full run remains required.
- Budget guard: E1–E4 and E5 reserve up to 20% for search return and proof/fallback work; final 50ms preflight on the sampled position used at most 40.6ms for E1–E5 (E1 1–5ms; E2 40.1ms; E3 40.2ms; E4 40.1ms; E5 40.6ms). A/B/C/D keep their V5.1 engine settings unchanged. D showed a 5s result 1.95ms over its cap in an earlier run; timeout is recorded as true when actual elapsed exceeds budget.
- Task 1 implementation frozen pending commit and V6 official tournament start.
