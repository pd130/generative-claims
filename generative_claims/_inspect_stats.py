import json
with open('data/processed/statistics.json') as f:
    stats = json.load(f)
cols = stats['_meta']['columns']
print('Columns:', cols)
print()
m = stats['columns']['model']
print('Models:', list(m['value_frequencies'].keys()))
print()
ft = stats['columns']['fuel_type']
print('Fuel types:', list(ft['value_frequencies'].keys()))
print()
s = stats['columns']['segment']
print('Segments:', list(s['value_frequencies'].keys()))
print()
r = stats['columns']['region_code']
print('Regions:', sorted(r['value_frequencies'].keys()))
print()
for col in ['customer_age', 'vehicle_age', 'subscription_length']:
    c = stats['columns'][col]
    print(col, ': min=', c['min'], ' max=', c['max'], ' mean=', round(c.get('mean', 0), 2))
print()
# binary/categorical columns
for col in cols:
    c = stats['columns'][col]
    if c.get('is_binary'):
        print('Binary:', col)
    elif c.get('type') == 'categorical' and col not in ('model', 'fuel_type', 'segment', 'region_code'):
        vf = c.get('value_frequencies', {})
        print(f'Cat: {col} → {list(vf.keys())[:5]}')
