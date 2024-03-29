# prj-comp-signate-1247-smbc
## 使用方法
- `docker-compose up`で環境作成
- scripts/下のコードが実行コード
   - 1ファイル1実験の構成

## 記録

| no | hash | CV | memo |
| -- | ---- | -- | ---- |
| 20231220_01 | 7dd636817dcfc21672f46cab1fdccce6fb7cbdbb | 0.3521835219475351 | EDA |
| 20231221_01 | 3245617eb98e850345d14ca30f29d0acb0d61f44 | 0.3561569724844358 | Fix categorical columns |
| 20231222_01 | 5e8df62ca7ddbc00985f37465d10299cd7458137 | 0.3554532007480837 | Drop duplicated columns |
| 20231222_02 | 2003eeac14f67b78aec550ff77621cd0ce06b037 | 0.4389039714606793 | Add target encoding (leaked) |
| 20231223_01 | 77de53b6a0287272838cdb9dd839220e371aeafd | 0.3554532007480837 | Check CV |
| 20231223_02 | 4a6a9f94ca720905d50905ad11073787745be0d0 | 0.3565386723074202 | Optimize prediction proba |
| 20231223_03 | c5bf7eaeeeb21ab1f8d8ccc682277703590c17df | 0.3637053181139169 | Change KFold to StratifiedKFold |
| 20231224_01 | 3adeea3cd1bff80d1284d8b46d21b5a74d7926f4 | 0.3419952132933971 | Try binary model for each class |
| 20231224_02 | e8c26e904fd136fec808038197a11e0299644145 | 0.3183868307681062 | Try regression model |
| 20231225_01 | db1be2f5cc1132f0448398870a74a55ec7bc4bdb | 0.3402735781884221 | Fix regression model |
| 20231226_01 | 66396b933f018e2a827a74c46fcb3668dce36e4e | 0.3476338994633227 | Try 3 binary model w/ regression opt |
| 20231226_02 | 78d0697361c37d031d2195903137009f009f7f10 | 0.3466229901621926 | Try to fix target encoding leakage |
| 20231227_01 | bd2bea5bd88c4de2549ac622a3d1e66b0bb3e280 | 0.3685119575375633 | Add static features |
| 20231227_02 | 94909033ee7cb91e86735b910cb03a571a9f4e08 | 0.3895433962532364 | Try 3 binary model + static features -> stack (leaked) |
| 20231228_01 | 3b531176e643be58c142238f9eedf25f02094137 | 0.3653823426970521 | Add target encoding |
| 20231228_02 | 6899ef9ff797b92286890520c60e73ae18c65047 | 0.366415570496963  | Add single column feature engineering |
| 20231229_01 | 08c6a126a7fb91851aeaf856697af2d9bc711b56 | 0.3515051702377618 | Try regression model and pseudo labeling (not worked, just copied pred.) |
| 20231230_01 | 0219be8eb28757ea66f179a102be38a9a19e18b4 | 0.3405435854405206 | Try 3 binary model w/ rule merging |
| 20231231_01 | 1c9a108dcb03f0e25b607d65ccbfdd2d0e7a8337 | 0.3674779680478224 | Add single column feature |
| 20231231_02 | 52d5c1e0514ee66d105d9c3c1399cfd0943fd7fe | 0.3579469806763103 | Add more quantitative feature |
| 20240101_01 | 930213c9a24b8d04f2d2e43194c5c591b71b90cb | 0.3720265447276095 | Try to optimize weight and loss |
