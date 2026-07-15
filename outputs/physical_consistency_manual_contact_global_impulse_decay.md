# Physical consistency — global impulse decay

Force 5.0 dataset/SAPIEN units; pulse 0.10 s; joint damping 2.0; joint friction 0.30; gravity disabled; 30 fps.

A maximum-duration stop is a hard failure; visual review cannot override it.

| ID | Object | Peak |qdot| | Final |qdot| | Settled | Duration verdict | Acceptance |
|---:|---|---:|---:|---|---|---|
| 10211 | Laptop | 0.0852819 | 1.42786e-10 | True | PASS_SETTLED_PLUS_HOLD | PASS |
| 10905 | Refrigerator | 0.03457 | 8.39497e-05 | True | PASS_SETTLED_PLUS_HOLD | PASS |
| 11100 | Scissors | 0.0264773 | 7.6774e-05 | True | PASS_SETTLED_PLUS_HOLD | PASS |
| 45135 | StorageFurniture | 0.00877094 | 9.34645e-05 | True | PASS_SETTLED_PLUS_HOLD | PASS |
| 100109 | USB | 0.00482072 | 9.77671e-05 | True | PASS_SETTLED_PLUS_HOLD | PASS |
| 101917 | Oven | 0.0192737 | 9.03181e-05 | True | PASS_SETTLED_PLUS_HOLD | PASS |
| 102255 | FoldingChair | 0.0349538 | 5.75465e-05 | True | PASS_SETTLED_PLUS_HOLD | PASS |
| 103111 | Stapler | 0.0111939 | 9.80329e-05 | True | PASS_SETTLED_PLUS_HOLD | PASS |
| 103706 | Knife | 0.0614376 | 4.89317e-05 | True | PASS_SETTLED_PLUS_HOLD | PASS |
| 103776 | WashingMachine | 0.278276 | 8.4184e-11 | True | PASS_SETTLED_PLUS_HOLD | PASS |

All 10 JSON files parse, contain finite numeric data, retain the fixed global parameters, record true pulse-only external forces, and link an MP4.
