# CVPR 2027 submission requirements (verified where possible)

Official pages (cvpr.thecvf.com) could not be fetched from this session; the items below come from web-search
snippets of the official CVPR 2027 Call for Papers / Dates pages and from the official author kit repository.
Items marked (unverified) must be re-checked on cvpr.thecvf.com before submission.

| Requirement | Status |
|---|---|
| Paper registration deadline: 10 Nov 2026; paper deadline: 16 Nov 2026 AoE; supplementary: 23 Nov 2026 | from official Dates page snippet |
| Reviews released 25 Jan 2027; decisions 25 Feb 2027; conference 22 to 25 June 2027 (Seattle) | from official snippet |
| Page limit: 8 pages including figures and tables; extra pages only for references and the mandatory LLM Usage Statement | from official CFP snippet |
| Papers exceeding the limit, not anonymised, not using the template, or lacking the LLM statement are rejected without review | from official CFP snippet |
| Submission through OpenReview; up-to-date OpenReview profiles for all co-authors; at most 25 papers per author | from official CFP snippet |
| Exact wording / placement of the LLM Usage Statement | **unverified**: one snippet says mandatory, another says the LLM policy is "being finalised"; the manuscript includes a statement after the references |
| LaTeX template: official author kit (github.com/cvpr-org/author-kit); the CVPR 2027 kit was not yet tagged at the time of writing; the latest master (commit 291758547e92, 27 May 2026, CVPR 2026 version) is used with `\confYear{2027}` | cvpr.sty, ieeenat_fullname.bst unmodified |
| Review version: `\usepackage[review]{cvpr}`, line numbers, anonymous author block | done |
| Dual submission / arXiv policy | **unverified** for 2027 |

Current manuscript (verified by compiling with the review option): pages 1 to 8 contain abstract through conclusion, pages 9 to 10 contain only references and the LLM Usage Statement;
then the LLM Usage Statement. Supplementary material compiles separately (`paper/supplementary.tex`).
