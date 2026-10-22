import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {
  FEATURE_NAMES, validateContext, buildFeatureVector, predictForest, budgetMetrics,
} from '../web/model.mjs';

const cold = () => ({amount: '10.00', count: '0', total: '0.00', unique: '0', isNewReceiver: true});
const prior = () => ({amount: '10.00', count: '2', total: '5.00', unique: '2', isNewReceiver: false});
const model = trees => ({schema_version: 1, model_name: 'forest_history', feature_names: FEATURE_NAMES, trees});
const leaf = value => ({left: [-1], right: [-1], feature: [-2], threshold: [-2], value: [value]});
const json = relative => JSON.parse(fs.readFileSync(new URL(relative, import.meta.url), 'utf8'));

test('decimal form strings normalize to an explicit context', () => {
  assert.deepEqual(validateContext(prior()), {
    amount: 10, count: 2, total: 5, unique: 2, isNewReceiver: false,
  });
  assert.deepEqual(validateContext({amount: 10, count: 2, total: 5, unique: 2, isNewReceiver: false}),
                   validateContext(prior()));
});

test('cold start has no invented history or amount ratio', () => {
  assert.deepEqual(buildFeatureVector(cold()), [Math.log1p(10), 0, 0, 0, 0, 1, 0]);
});

test('history summaries produce seven hand-computed features in training order', () => {
  assert.deepEqual(buildFeatureVector(prior()), [Math.log1p(10), 2, 5, 2.5, 2, 0, 4]);
  assert.deepEqual(FEATURE_NAMES, [
    'log_amount', 'sender_count_24h', 'sender_total_24h', 'sender_mean_24h',
    'sender_unique_receivers_24h', 'is_new_receiver_24h', 'amount_to_prior_mean_ratio',
  ]);
});

test('one-cent histories and documented money/count limits remain valid', () => {
  const cents = validateContext({...prior(), count: 57, total: '0.57', unique: 1});
  assert.equal(cents.total, 0.57);
  const maximum = {...prior(), amount: '99999999999.99', total: '99999999999.99', count: 10000, unique: 10000};
  assert.ok(buildFeatureVector(maximum).every(Number.isFinite));
});

test('money accepts only finite plain decimal strings or numbers', () => {
  for (const amount of ['', '1e3', 'NaN', 'Infinity', '-1', '1.001', '100000000000',
                        NaN, Infinity, -1, 0, true, null, {}, [], ['1.00']]) {
    assert.throws(() => validateContext({...cold(), amount}), `accepted invalid amount ${String(amount)}`);
  }
});

test('context rejects inconsistent counts, receivers, totals and nonboolean flags', () => {
  const changes = [
    {count: -1}, {count: 1.5}, {count: 10001}, {count: '1e2'}, {count: [2]},
    {unique: 3}, {unique: 0}, {unique: 1.5}, {total: 0}, {total: '0.01'},
    {isNewReceiver: 'false'}, {isNewReceiver: 1},
  ];
  for (const change of changes) assert.throws(() => validateContext({...prior(), ...change}));
  for (const change of [{total: 1}, {unique: 1}, {isNewReceiver: false}]) {
    assert.throws(() => validateContext({...cold(), ...change}));
  }
  for (const input of [null, undefined, '10', 5]) assert.throws(() => validateContext(input));
});

test('forest averages leaf values and casts inputs to float32 before comparison', () => {
  assert.equal(predictForest(model([leaf(0.2), leaf(0.8)]), [1, 0, 0, 0, 0, 1, 0]), 0.5);
  const tree = {left: [1, -1, -1], right: [2, -1, -1], feature: [0, -2, -2],
                threshold: [1.00000003, -2, -2], value: [0.5, 0.25, 0.75]};
  assert.ok(1.00000004 > tree.threshold[0]);
  assert.ok(Math.fround(1.00000004) <= tree.threshold[0]);
  assert.equal(predictForest(model([tree]), [1.00000004, 0, 0, 0, 0, 1, 0]), 0.25);
  assert.equal(predictForest(model([tree]), [1.0000002, 0, 0, 0, 0, 1, 0]), 0.75);
});

test('inference rejects malformed vectors, assets, invalid leaves and cyclic trees', () => {
  const valid = model([leaf(0.5)]), vector = [1, 0, 0, 0, 0, 1, 0];
  for (const features of [[], Array(7), vector.slice(1), [NaN, ...vector.slice(1)],
                           [Infinity, ...vector.slice(1)], [-1, ...vector.slice(1)],
                           ['1', ...vector.slice(1)], [1e100, ...vector.slice(1)]]) {
    assert.throws(() => predictForest(valid, features));
  }
  for (const invalid of [null, {...valid, schema_version: 2}, {...valid, model_name: 'other'},
                         {...valid, feature_names: [...FEATURE_NAMES].reverse()}, model([]),
                         model([leaf(NaN)]), model([leaf(1.1)]),
                         model([{...leaf(0.5), right: []}]),
                         model([{left: [0], right: [0], feature: [0], threshold: [1], value: [0.5]}])]) {
    assert.throws(() => predictForest(invalid, vector));
  }
});

test('all 80 synthetic threshold vectors match the saved sklearn reference', () => {
  const forest = json('../web/data/forest.json');
  const parity = json('./fixtures/web_parity.json');
  const report = json('../web/data/report.json');
  const forestBytes = fs.readFileSync(new URL('../web/data/forest.json', import.meta.url));
  const forestHash = createHash('sha256').update(forestBytes).digest('hex');
  assert.equal(parity.cases.length, 80);
  assert.equal(parity.source_model_sha256, forest.source_model_sha256);
  assert.equal(parity.forest_json_sha256, forestHash);
  assert.equal(report.web_export.forest_json_sha256, forestHash);
  assert.equal(report.web_export.source_model_sha256, forest.source_model_sha256);
  assert.equal(report.web_export.training_input_sha256, forest.training_input_sha256);
  assert.equal(report.input_sha256, forest.training_input_sha256);
  for (const [index, entry] of parity.cases.entries()) {
    assert.equal(entry.features.length, 7);
    const actual = predictForest(forest, entry.features);
    assert.ok(Math.abs(actual - entry.expected) < 1e-12, `reference mismatch at case ${index}`);
  }
});

test('review budget lookup uses the actual published fixed-budget counts', () => {
  const report = json('../web/data/report.json');
  assert.equal(Object.hasOwn(report, 'artifacts'), false, 'Local artifact paths must not be published.');
  assert.deepEqual(report.selected_review_budgets.map(entry => entry.budget), [.01, .02, .05, .1, .2]);
  for (const entry of report.selected_review_budgets) {
    assert.equal(budgetMetrics(report, entry.budget), entry);
    assert.equal(entry.review_count, Math.ceil(entry.rows * entry.budget));
    assert.equal(entry.true_positives + entry.false_positives, entry.review_count);
    assert.equal(entry.true_positives + entry.false_negatives, entry.positives);
    assert.equal(entry.precision, entry.true_positives / entry.review_count);
    assert.equal(entry.recall, entry.true_positives / entry.positives);
  }
  const original = budgetMetrics(report, .05);
  assert.equal(original.review_count, 2037);
  assert.equal(original.true_positives, 160);
  assert.equal(original.false_positives, 1877);
  assert.equal(original.false_negatives, 162);
  assert.equal(original.average_precision, report.test.forest_history.average_precision);
});

test('unsupported or malformed review budgets fail explicitly', () => {
  const report = {selected_review_budgets: [{budget: .05, review_count: 1}]};
  for (const budget of [0, -1, .03, .051, 1, NaN, Infinity, '0.05', null]) {
    assert.throws(() => budgetMetrics(report, budget));
  }
  for (const invalid of [null, {}, {selected_review_budgets: null}]) {
    assert.throws(() => budgetMetrics(invalid, .05));
  }
});
