# Final Orbit-GOP public release audit

Audit date: 2026-09-28 (Asia/Shanghai). Scope: the local `GITHUB_RELEASE_PUBLIC_CANDIDATE` directory and its matching ZIP. No GitHub push, Release, DOI, new training, or original Bootstrap was performed.

## Decision

| Item | Final assessment |
|---|---|
| Public candidate | **READY** |
| Public release technically ready | **YES** |
| Human authorization | **APPROVED** for public disclosure of this exact candidate |
| Project policy | **SOURCE-AVAILABLE RESEARCH RELEASE** |
| Permissive OSS license for original Orbit-GOP contributions | **NO** |
| Non-commercial academic review and verification | **PERMITTED** under `USAGE_NOTICE.md` |
| Commercial use, redistribution, or distribution of modified versions without written permission | **NO** |
| Third-party license check | **CONDITIONAL** |
| Scientific content freeze | **PASS** |
| R0 / R1 / R2 | **VERIFIED** within the documented release checks |
| R3 | **NOT PROVIDED BY THIS RELEASE** |

Public disclosure of this release has been approved by the project rights holder(s). `RELEASE_AUTHORIZATION_STATUS.md`, `USAGE_NOTICE.md`, README, and `docs/PROJECT_LICENSE_STATUS.md` record the approved scope and limited original-code permission. The notice expressly excludes third-party portions from its original-contribution grant. The current candidate has no operative `LICENSE_PENDING.md` or authorization-required document. The historical exclusion inventory retains the old filename only as a record of its removal.

## Severity reassessment

| Level | Count | Basis |
|---|---:|---|
| P0 | 0 | No verified credential or private path disclosure, frozen-content change, missing third-party notice text, or package hash mismatch. |
| P1 | 0 | The prior human-authorization decision is now approved and recorded for this exact candidate. Historical archive executors remain excluded. |
| P2 | 0 | No unresolved candidate-scope defect found in this pass. |
| P3 | 2 | Historical figure/data filenames remain stable; the standalone `q_func` empty-class API boundary is narrower than the formal evaluator's used-class behavior. Neither changes the frozen S3–S5 path. |

## Evidence and limits

- The original `GITHUB_RELEASE_FINAL` directory has 352 files and matches its prior file-level SHA256 baseline with 0 changes. The scientific code, configurations, splits, results, and tests in this candidate are byte-identical to the pre-edit candidate archive. Only release policy and audit documents were changed; `provenance/CODE_AUDIT.csv` was updated solely in policy-status fields, preserving each source path, classification, and source/candidate digest. See `FINAL_SCIENTIFIC_FREEZE_CHECK.md`.
- Four third-party license texts remain byte-identical. C-EDL and OpenOOD MIT notice hashes match the existing local upstream-verification record. GradNorm Apache-2.0 attribution and torchvision BSD-3-Clause text remain present. Exact remote upstream revisions were not independently reverified in this pass, and the exact torchvision source commit remains unknown; therefore the attribution check is **CONDITIONAL**, with those limits disclosed in `THIRD_PARTY_NOTICES.md`.
- The complete candidate was scanned for the requested credential, private-path, and internal-dialogue markers. No actionable disclosure or credential file was found. The only routine matches were environment-variable code and ignored `.env` patterns, plus `PRIVATE_DESKTOP` provenance aliases in figure source maps; those aliases are not real user-home paths. The candidate has no original model weights, per-sample scores, raw datasets, or full historical executors.
- All 55 Python files parsed successfully; 6 synthetic CPU and release-safety unit tests passed. These checks do not re-execute CUDA training, evaluate the original nine weights, or rerun the 5,000-replicate Bootstrap.
- The README and reproducibility record retain R0/R1/R2 **VERIFIED** and R3 **NOT PROVIDED BY THIS RELEASE**. The release does not claim exact full-paper regeneration or journal acceptance.
- `MANIFEST.csv` indexes each payload file except the two index files; `SHA256SUMS.txt` covers all files except itself, including the manifest. The final ZIP contains the same files as the directory. The ZIP SHA256 is reported separately with the handoff because a ZIP cannot contain its own digest without a circular dependency.

Final local release decision: **READY FOR GITHUB PUBLIC = YES**, subject to the documented source-available terms and third-party license boundaries. This is a local package decision; no remote publication operation was performed.
