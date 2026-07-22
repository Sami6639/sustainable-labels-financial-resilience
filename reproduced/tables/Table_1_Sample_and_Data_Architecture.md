### Table 1. Sample and data architecture

| Layer                          | Period / snapshot        | Coverage                              | Purpose                             |
|:-------------------------------|:-------------------------|:--------------------------------------|:------------------------------------|
| Monthly ETF panel              | Jan. 2010–Dec. 2025      | 4,459 ETF-months; 50 ETFs; 192 months | Pricing and recovery                |
| Fixed sustainable ETF universe | 2025Q4 classification    | 72 included equity ETFs               | Predefined historical reference set |
| Historical N-PORT match        | 2023Q4 / 2024Q4 / 2025Q4 | 56 / 63 / 66 exact series matches     | Holdings reconstruction             |
| Complete architecture panel    | All three snapshots      | 56 ETFs with valid scores             | Persistence and persistent averages |
| Equity-corporate holdings      | 2023Q4 / 2024Q4 / 2025Q4 | 15,742 / 16,778 / 16,539 rows         | Portfolio weighting                 |
| Firm fundamentals              | FY2010–FY2025            | 27,312 firm-years; 2,174 firms        | Lagged firm characteristics         |
| Extreme CPU / joint stress     | Monthly panel            | 20 / 17 months                        | Dynamic and threshold states        |

*Note.* The fixed ETF universe is determined from the 2025Q4 review and applied backward. The complete architecture count requires valid scores in every snapshot.
