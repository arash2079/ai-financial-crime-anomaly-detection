import { validateContext, buildFeatureVector, predictForest } from './model.mjs';

const $ = (id) => document.getElementById(id);
const integer = new Intl.NumberFormat('en-US');
const percent = (value) => `${(value * 100).toFixed(2)}%`;
const modelNames = {
  amount_rule: 'Amount rule', logistic_amount: 'Logistic · amount',
  logistic_history: 'Logistic · history', forest_amount: 'Forest · amount',
  forest_history: 'Forest · history', isolation_history: 'Isolation · history',
};
const featureNames = [
  'log_amount', 'sender_count_24h', 'sender_total_24h', 'sender_mean_24h',
  'sender_unique_receivers_24h', 'is_new_receiver_24h', 'amount_to_prior_mean_ratio',
];
const scenarios = [
  { amount: '1250.00', count: '0', total: '0', unique: '0', isNewReceiver: true },
  { amount: '9500.00', count: '4', total: '4200.00', unique: '2', isNewReceiver: false },
  { amount: '50000.00', count: '3', total: '9000.00', unique: '3', isNewReceiver: true },
];
let report = null;
let forest = null;
let modelLoading = null;

async function fetchJson(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`Unable to load ${path} (${response.status}).`);
  return response.json();
}

function activateTab(tab, focus = false) {
  document.querySelectorAll('[role="tab"]').forEach((item) => {
    const selected = item === tab;
    item.setAttribute('aria-selected', String(selected));
    item.tabIndex = selected ? 0 : -1;
    $(item.getAttribute('aria-controls')).hidden = !selected;
  });
  if (focus) tab.focus();
  if (tab.id === 'tab-try') loadModel();
}

const tabs = [...document.querySelectorAll('[role="tab"]')];
tabs.forEach((tab, index) => {
  tab.addEventListener('click', () => activateTab(tab));
  tab.addEventListener('keydown', (event) => {
    let target;
    if (event.key === 'ArrowRight') target = (index + 1) % tabs.length;
    if (event.key === 'ArrowLeft') target = (index - 1 + tabs.length) % tabs.length;
    if (event.key === 'Home') target = 0;
    if (event.key === 'End') target = tabs.length - 1;
    if (target !== undefined) {
      event.preventDefault();
      activateTab(tabs[target], true);
    }
  });
});
$('hero-try').addEventListener('click', () => {
  activateTab($('tab-try'), true);
  $('tab-try').scrollIntoView({ block: 'start' });
});

function renderReport(data) {
  if (!data.test || !data.validation || !data.test[data.selected_model]) {
    throw new Error('The measured report has an unsupported format.');
  }
  report = data;
  const chosen = data.test[data.selected_model];
  $('headline-recall').textContent = percent(chosen.recall);
  $('headline-precision').textContent = percent(chosen.precision);
  $('headline-false').textContent = integer.format(chosen.false_positives);
  const body = $('comparison-body');
  body.replaceChildren();
  Object.entries(modelNames).forEach(([name, label]) => {
    const test = data.test[name];
    const validation = data.validation[name];
    if (!test || !validation) throw new Error('The measured report is incomplete.');
    const row = document.createElement('tr');
    const title = document.createElement('th');
    title.scope = 'row';
    title.textContent = label;
    if (name === data.selected_model) {
      row.className = 'selected-row';
      const badge = document.createElement('span');
      badge.className = 'selected-badge';
      badge.textContent = 'SELECTED';
      title.append(badge);
    }
    row.append(title);
    [validation.average_precision.toFixed(6), test.average_precision.toFixed(6),
      percent(test.precision), percent(test.recall)].forEach((value) => {
      const cell = document.createElement('td');
      cell.textContent = value;
      row.append(cell);
    });
    body.append(row);
  });
  $('report-status').textContent = 'Measured on the fixed September 9–10, 2022 test period.';
  renderBudget();
}

function renderBudget() {
  if (!report) return;
  const budget = Number($('budget').value);
  const metrics = report.selected_review_budgets?.find((item) => Math.abs(item.budget - budget) < 1e-10);
  if (!metrics) {
    $('budget-summary').textContent = 'Review-budget results are unavailable.';
    return;
  }
  $('budget-reviewed').textContent = integer.format(metrics.review_count);
  $('budget-found').textContent = integer.format(metrics.true_positives);
  $('budget-false').textContent = integer.format(metrics.false_positives);
  $('budget-missed').textContent = integer.format(metrics.false_negatives);
  $('budget-summary').textContent = `Recall ${percent(metrics.recall)} · Precision ${percent(metrics.precision)}${budget === 0.05 ? ' · Frozen primary budget.' : ' · Descriptive exploration only.'}`;
}
$('budget').addEventListener('change', renderBudget);

async function loadModel() {
  if (forest) return forest;
  if (modelLoading) return modelLoading;
  $('run-model').disabled = true;
  $('retry-model').hidden = true;
  $('model-status').textContent = 'Downloading the trained model for local inference…';
  modelLoading = fetchJson('./data/forest.json').then((data) => {
    if (!data.trees?.length || data.model_name !== 'forest_history') {
      throw new Error('The model file has an unsupported format.');
    }
    forest = data;
    $('run-model').disabled = false;
    $('model-status').textContent = 'Model ready. Predictions run locally in this browser.';
    return data;
  }).catch(() => {
    $('model-status').textContent = 'The model could not be downloaded. Check your connection and retry.';
    $('retry-model').hidden = false;
    return null;
  }).finally(() => { modelLoading = null; });
  return modelLoading;
}
$('retry-model').addEventListener('click', loadModel);

function readContext() {
  return {
    amount: $('amount').value, count: $('count').value,
    total: $('total').value, unique: $('unique').value,
    isNewReceiver: $('is-new').checked,
  };
}

function syncHistory() {
  const empty = /^0+$/.test($('count').value.trim());
  $('total').disabled = empty;
  $('unique').disabled = empty;
  $('is-new').disabled = empty;
  if (empty) {
    $('total').value = '0';
    $('unique').value = '0';
    $('is-new').checked = true;
  }
}

function invalidateScore() {
  $('score-value').textContent = '—';
  $('score-message').textContent = 'Inputs changed. Run the model to calculate a new score.';
  $('derived-details').hidden = true;
  $('form-error').hidden = true;
}

$('model-form').addEventListener('input', (event) => {
  if (event.target.id === 'count') syncHistory();
  invalidateScore();
});
$('scenario').addEventListener('change', () => {
  const scenario = scenarios[Number($('scenario').value)];
  ['amount', 'count', 'total', 'unique'].forEach((key) => { $(key).value = scenario[key]; });
  $('is-new').checked = scenario.isNewReceiver;
  syncHistory();
  invalidateScore();
});

$('model-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  $('form-error').hidden = true;
  try {
    const context = validateContext(readContext());
    const features = buildFeatureVector(context);
    if (!forest && !await loadModel()) return;
    const score = predictForest(forest, features);
    $('score-value').textContent = score.toFixed(4);
    $('score-message').textContent = 'This score describes only this model’s ranking of the supplied context. It does not determine whether a crime occurred.';
    const derived = $('derived-features');
    derived.replaceChildren();
    features.forEach((value, index) => {
      const key = document.createElement('dt');
      const output = document.createElement('dd');
      key.textContent = featureNames[index];
      output.textContent = new Intl.NumberFormat('en-US', { maximumFractionDigits: 4 }).format(value);
      derived.append(key, output);
    });
    $('derived-details').hidden = false;
  } catch (error) {
    $('score-value').textContent = '—';
    $('derived-details').hidden = true;
    $('form-error').textContent = error instanceof Error ? error.message : 'Please check the transaction context.';
    $('form-error').hidden = false;
    $('score-message').textContent = 'No score was calculated. Correct the inputs and try again.';
  }
});

syncHistory();
fetchJson('./data/report.json').then(renderReport).catch(() => {
  $('report-status').textContent = 'The results file could not be loaded. The headline values above are the recorded primary result; reload to retry the comparison table.';
  $('budget-summary').textContent = 'The measured review-budget results could not be loaded.';
});
