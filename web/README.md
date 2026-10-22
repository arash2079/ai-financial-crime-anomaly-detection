# Interactive demo

A small buildless HTML/CSS/JavaScript app for GitHub Pages, not a hosted
prediction service. The two sections explain the experiment and let visitors
change a hypothetical USD ACH payment and its historical context.

![Locally tested browser demo](../docs/demo-preview.jpg)

Screenshot from a local browser test, not a public-deployment confirmation.

## What runs

`data/forest.json` contains the **same selected 100-tree forest** trained in
the frozen comparison, not a new model or a handcrafted scoring rule.
`model.mjs` constructs the seven numerical inputs, casts them to float32 as
sklearn does, follows each tree's <= branches and averages positive-class leaf
fractions. Class weighting makes this a ranking score, not a calibrated risk
probability. The app gives no laundering verdict or approval decision.

The form accepts current amount, prior outgoing count/total, distinct receivers
and whether the receiver is new. It derives mean, amount ratio and log amount.
Historical context must refer only to earlier outgoing ACH/USD payments within
24 hours. The app does not reconstruct history from bank uploads. Manual entries
and the three handcrafted examples are not Kaggle records and have no labels.
Some hypothetical contexts may fall outside training data. Input bounds are
interface safeguards, not learned bounds or compliance rules.

The budget explorer reads aggregate results computed from the saved test
scores. The primary comparison remains 5%; 1/2/10/20% are descriptive views
added afterward, not model tuning. They are independent of the form's scenario.
Top-k reviews over an entire period are not an online decision threshold.

Only static assets are fetched. Form values are not saved, uploaded or included
in URLs. No analytics, external fonts, CDNs, login, database or prediction API.
GitHub still serves ordinary website requests; browser-local inference is not
a promise of anonymous web browsing.

## Rebuild and checks

After the data preparation and training steps in the main README:

```bash
python -m src.export_web
python -m unittest discover -s tests -v
node --test tests/test_web.mjs
```

The exporter reads only the project's own trusted joblib artifact; never supply
untrusted model files. Export does not fit or tune anything. It verifies saved
test scores against the estimator, records the model/input/export SHA-256 hashes,
and removes local filesystem paths from the public report. The approximately
1.43 MB model is loaded when the interactive section opens.

80 synthetic feature vectors test thresholds and float32 neighbors against
sklearn reference scores. These are numerical parity fixtures, not exported
transaction records. Additional tests cover money/count validation, empty
histories, invalid trees, fixed-budget counts and exporter integrity.

Desktop browser checks covered both tabs, keyboard navigation, scenario changes,
real inference, derived features, invalid inputs, stale-score clearing and
budget changes. Responsive CSS is included; a physical-phone test is not yet
recorded. Loading failures have a retry state. A separate prediction server is
not required.

## Publish

In **this repository**, set Settings → Pages → Build and deployment → Source
to **GitHub Actions**. The workflow tests Python and JavaScript, then deploys
only `web/` on pushes to main or manual workflow dispatch. Pull requests run
checks without deployment. The `github-pages` environment and minimal Pages/OIDC
permissions are scoped to deployment. Any protected-environment approval remains
an owner action; the workflow does not bypass it.

GitHub supplies the live URL in the successful deployment step. A committed
workflow or passing tests alone do not mean the website is deployed. If first
configuration was missing, enable it and re-run the failed deployment job.

Code is MIT. The model was trained on IBM's synthetic Kaggle data; source
attribution, version and separate CDLA-Sharing-1.0 data license are documented
in [data notes](../data/README.md). No raw transactions or account IDs are served.
