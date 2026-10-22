// Small, dependency-free inference for the exported sklearn Random Forest.
export const FEATURE_NAMES = [
  'log_amount', 'sender_count_24h', 'sender_total_24h', 'sender_mean_24h',
  'sender_unique_receivers_24h', 'is_new_receiver_24h', 'amount_to_prior_mean_ratio',
];

function money(value, label) {
  if (!['string', 'number'].includes(typeof value)) throw new Error(`${label}: enter a USD amount.`);
  const text = String(value ?? '').trim();
  if (!/^(0|[1-9]\d{0,10})(\.\d{1,2})?$/.test(text)) {
    throw new Error(`${label}: enter a plain USD amount with at most two decimal places.`);
  }
  const [whole, fraction = ''] = text.split('.');
  const cents = Number(whole) * 100 + Number(fraction.padEnd(2, '0'));
  if (!Number.isSafeInteger(cents)) throw new Error(`${label} is too large.`);
  return cents / 100;
}

function integer(value, label) {
  if (!['string', 'number'].includes(typeof value)) throw new Error(`${label}: enter a whole number.`);
  const text = String(value ?? '').trim();
  if (!/^(0|[1-9]\d{0,4})$/.test(text) || Number(text) > 10000) {
    throw new Error(`${label}: enter a whole number from 0 to 10,000.`);
  }
  return Number(text);
}

export function validateContext(input) {
  if (!input || typeof input !== 'object') throw new Error('Enter a transaction context.');
  const amount = money(input.amount, 'Current amount');
  const total = money(input.total, 'Prior total');
  const count = integer(input.count, 'Prior count');
  const unique = integer(input.unique, 'Distinct prior receivers');
  const isNewReceiver = input.isNewReceiver;
  if (typeof isNewReceiver !== 'boolean') throw new Error('New receiver must be true or false.');
  if (amount <= 0) throw new Error('Current amount must be greater than zero.');
  if (unique > count) throw new Error('Distinct receivers cannot exceed the prior payment count.');
  if (count === 0 && (total !== 0 || unique !== 0 || !isNewReceiver)) {
    throw new Error('With no prior payments, total and receivers must be zero and the receiver must be new.');
  }
  if (count > 0 && (Math.round(total * 100) < count || unique < 1)) {
    throw new Error('Prior payments need a positive total (at least one cent each) and at least one receiver.');
  }
  return { amount, count, total, unique, isNewReceiver };
}

export function buildFeatureVector(input) {
  const { amount, count, total, unique, isNewReceiver } = validateContext(input);
  const mean = count ? total / count : 0;
  return [Math.log1p(amount), count, total, mean, unique, Number(isNewReceiver), mean ? amount / mean : 0];
}

export function predictForest(model, features) {
  if (!model || model.schema_version !== 1 || model.model_name !== 'forest_history' ||
      JSON.stringify(model.feature_names) !== JSON.stringify(FEATURE_NAMES) ||
      !Array.isArray(model.trees) || !model.trees.length || model.trees.length > 500) {
    throw new Error('The model asset has an unsupported format.');
  }
  if (!Array.isArray(features) || features.length !== FEATURE_NAMES.length ||
      Array.from(features).some(value => typeof value !== 'number' || !Number.isFinite(value) || value < 0)) {
    throw new Error('Inference requires seven finite, nonnegative features.');
  }
  // sklearn converts prediction input to float32 before comparing thresholds.
  const values = features.map(Math.fround);
  if (values.some(value => !Number.isFinite(value))) throw new Error('Features exceed the model numerical range.');
  let sum = 0;
  for (const tree of model.trees) {
    const arrays = ['left', 'right', 'feature', 'threshold', 'value'];
    if (!tree || arrays.some(key => !Array.isArray(tree[key])) || !tree.left.length ||
        arrays.some(key => tree[key].length !== tree.left.length)) {
      throw new Error('The model contains an invalid tree.');
    }
    let node = 0;
    let found = false;
    for (let step = 0; step < tree.left.length; step++) {
      if (!Number.isInteger(node) || node < 0 || node >= tree.left.length) break;
      const left = tree.left[node], right = tree.right[node];
      if (left === -1 && right === -1) {
        const value = tree.value[node];
        if (typeof value !== 'number' || !Number.isFinite(value) || value < 0 || value > 1) break;
        sum += value;
        found = true;
        break;
      }
      const feature = tree.feature[node], threshold = tree.threshold[node];
      if (!Number.isInteger(feature) || feature < 0 || feature >= values.length ||
          typeof threshold !== 'number' || !Number.isFinite(threshold)) break;
      node = values[feature] <= threshold ? left : right;
    }
    if (!found) throw new Error('The model contains an invalid or cyclic tree.');
  }
  return sum / model.trees.length;
}

export function budgetMetrics(report, budget) {
  if (typeof budget !== 'number' || !Number.isFinite(budget) || !Array.isArray(report?.selected_review_budgets)) {
    throw new Error('Review-budget data is unavailable.');
  }
  const result = report.selected_review_budgets.find(entry => entry.budget === budget);
  if (!result) throw new Error('Choose one of the published review budgets.');
  return result;
}
