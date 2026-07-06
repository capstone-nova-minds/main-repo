# Capstone Team Prepwork — Cheat Sheet

**Team Name:** Nova Minds  
**GitHub Organization:** <https://github.com/capstone-nova-minds>  
**Repository:** <https://github.com/capstone-nova-minds/main-repo>  
**Project Board:** <https://github.com/orgs/capstone-nova-minds/projects/3>

Kickoff meeting reference. Fill in updates live with the team.

---

## 1. Sprint Schedule — Fixed Deadlines

| Date | Milestone | Updates | Completed |
|---|---|---|---|
| Sun 5 Jul, 1:30 PM | Kickoff lecture for the whole cohort | Attended | ✅ |
| Sun 5 Jul | Team meeting: GitHub setup, brainstorm project ideas, write Team Contract | GitHub setup done | ✅ |
| Mon 6 Jul, 12:00 PM | Team Contract | done | ✅ |
| Mon 6 Jul PM → Tue 7 Jul | Refine project idea and write Project Proposal |  |  |
| Tue 7 Jul, 8:00 PM | Project Proposal due |  |  |
| Wed 8 Jul AM | Proposal approved or revision requested |  |  |
| Wed 8 Jul → Sun 12 Jul | Build Week 1 |  |  |
| Thu 9 or Sun 12 Jul | Progress Check Zoom — mandatory, all teammates attend |  |  |
| Mon 13 Jul → Tue 14 Jul | Build Week 2 |  |  |
| Tue 14 Jul, 8:00 PM | All technical work due: Final PR, demo video, portfolio artifact, and deck |  |  |
| Wed 15 or Thu 16 Jul | Demo Day |  |  |
| Fri 17 Jul | Rest day |  |  |
| Sat 18 Jul, 8:00 PM | Individual Reflection + Peer Reflection due |  |  |


---

# Team Contract

**Due:** Mon 6 Jul, 12:00 PM Amman  
**File name:** `TEAM_CONTRACT.md`  
**Submission:** Commit the file to the team repository.

---

## 1. Team Roster

| Name | GitHub Handle / Profile | Contact Hours |
|---|---|---|
| Ibrahim Almomani | [Giddygit7](https://github.com/Giddygit7) | 10:00 AM–11:00 PM Amman |
| Omar Al-akhra | [omar2003has-creator](https://github.com/omar2003has-creator) | 10:00 AM–11:00 PM Amman |
| Rawan Quraish | [RawanQuraish](https://github.com/RawanQuraish) | 10:00 AM–11:00 PM Amman |
| Hadeel Banihani | [hadeelbanihani22-dotcom](https://github.com/hadeelbanihani22-dotcom) | 10:00 AM–11:00 PM Amman |

---

## 2. Cooperation Plan

| Team Member | Strengths | How We Will Use These Strengths |
|---|---|---|
| Hadeel | RAG systems, retrieval design, chatbot answer testing, accuracy improvement, documentation, and sprint development | Support the RAG pipeline, improve retrieval quality, test chatbot answers, and document the project clearly |
| Omar | AI/RAG system pipelines, Knowledge Graphs, Transformers, model fine-tuning with PyTorch, and production-ready AI data pipelines | Support the AI pipeline, Knowledge Graph design, model-related decisions, and system integration |
| Ibrahim | Backend development, data handling, evaluation, frontend development, and infrastructure integration | Support backend implementation, data processing, evaluation workflow, and integration tasks |
| Rawan | Frontend development, UI implementation, responsive design, and user-friendly interfaces | Lead UI implementation and ensure the demo is clean, responsive, and easy to use |

---

## 3. Conflict Plan

Disagreements will be raised directly and discussed within 24 hours in a short 15-minute team meeting. The team will not leave disagreements unresolved.

If a teammate misses two consecutive check-ins without notice, the team will first discuss the issue privately with the teammate. If the issue continues, the team will escalate it to the instructor.

If the team cannot reach consensus on a decision, the decision will be made by majority vote.

---

## 4. Communication Plan

| Item | Team Rule |
|---|---|
| Main communication platforms | Slack, WhatsApp, and Google Meet |
| Working hours | 10:00 AM–11:00 PM Amman |
| Response window | Team members should respond within a reasonable time during working hours |
| Meetings | Google Meet will be used for team calls and sprint check-ins |
| After-hours rule | Urgent messages can be sent, but responses are not expected immediately outside agreed working hours |

---

## 5. Work Plan

We will use the GitHub Project Board to track all tasks and progress.

Tasks will move through the following workflow:

```text
Backlog → In progress → In review → Done
```

### Definition of Done

A task is considered done only when:

- The assigned work is completed.
- The code or documentation is pushed to the correct branch.
- A Pull Request is opened when needed.
- Required reviewers check and approve the work.
- The change does not break existing functionality.
- Documentation is updated when needed.

### No-Solo-Committing Rule

No teammate should push directly to `main`. All important changes must go through Pull Requests and team review.

---

## 6. Git Process

### Repository and Board URLs

- **Team org URL:** <https://github.com/capstone-nova-minds>
- **Repo URL:** <https://github.com/capstone-nova-minds/main-repo>
- **Project board URL:** <https://github.com/orgs/capstone-nova-minds/projects/3>

### Branch Strategy

We will use `main` for clean and stable production-ready code. The `main` branch is protected.

Each teammate will work on a separate branch named after the task or feature, for example:

- `feature/data`
- `feature/backend`
- `feature/ui`
- `feature/rag-pipeline`
- `bugfix/issue-name`
- `docs/task-name`

No direct commits are allowed to `main`.

### Pull Request Review Workflow

When a task is completed, the teammate will open a Pull Request.

Each Pull Request into `main` requires at least **two approvals** from other team members before merging.

Reviewers will check:

- The code runs without errors.
- The task matches the GitHub Project card.
- No secrets, API keys, passwords, or `.env` files are committed.
- The change does not break existing functionality.
- Documentation is updated when needed.

### Merge Cadence

Completed and verified features will be merged into `main` frequently to avoid large conflicts.

The team will review Pull Requests daily during the sprint and keep everyone updated by running:

```bash
git pull
```

Large features should be split into smaller Pull Requests when possible.

### Branch Protection Rules

- The `main` branch is protected.
- Pull Requests are required before merging.
- Each Pull Request requires 2 approvals.
- Stale approvals are dismissed when new commits are pushed.
- Code Owner review is not required.
- The bypass list will remain empty.
- No teammate should bypass branch protection rules.

---

## 7. Presentation Practice

| Team Member | Presentation Role |
|---|---|
| Rawan | Leads the presentation narrative and ensures the story is clear from problem to solution, architecture, evaluation, demo, and lessons learned |
| Omar | Runs the live demo during Demo Day and prepares a backup demo video or screenshots in case the live demo does not work |
| Hadeel | Explains the technical part, including the AI capability, architecture, data flow, and evaluation results |
| Ibrahim | Coordinates the Q&A section and ensures each question is answered by the teammate responsible for that part |

### Rehearsal Cadence

The team will complete one full rehearsal one day before Demo Day.

The team will also complete one short rehearsal after the technical work is completed to test the demo flow, timing, and speaker transitions.

---

## 8. Sign-off

By checking our names below, we confirm that we participated in writing this Team Contract, reviewed its content, and agree to follow it during the Capstone sprint.

| Team Member | Confirmation |
|---|---|
| Hadeel | ✅ Confirmed |
| Rawan | ✅ Confirmed |
| Ibrahim | ✅ Confirmed |
| Omar | ✅ Confirmed |
