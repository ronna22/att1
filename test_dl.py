import urllib.request, urllib.parse, json, time, datetime

symbol = 'SIRENUSDT'
gran = '15m'
now_ms = int(time.time() * 1000)
start_ms = now_ms - 100 * 24 * 3600 * 1000  # 100 days back
current_end = now_ms
all_rows = []
calls = 0

while current_end > start_ms and calls < 15:
    params = {
        'symbol': symbol, 'granularity': gran, 'productType': 'USDT-FUTURES',
        'endTime': str(current_end), 'limit': '1000'
    }
    url = 'https://api.bitget.com/api/v2/mix/market/candles?' + urllib.parse.urlencode(params)
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.loads(r.read())
    except Exception as e:
        print(f'HTTP error on call {calls+1}: {e}')
        break
    code = data.get('code')
    if code != '00000':
        msg = data.get('msg', '')
        print(f'API error on call {calls+1}: code={code} msg={msg}')
        break
    rows = data.get('data', [])
    calls += 1
    newest = rows[0][0] if rows else 'N/A'
    oldest = rows[-1][0] if rows else 'N/A'
    print(f'Call {calls}: got {len(rows)} rows  oldest={oldest}  newest={newest}')
    if not rows:
        break
    for row in rows:
        ts = int(row[0])
        if ts >= start_ms:
            all_rows.append(ts)
    oldest_ts = int(rows[-1][0])
    if oldest_ts <= start_ms:
        break
    current_end = oldest_ts - 1
    time.sleep(0.15)

print(f'Total rows collected: {len(all_rows)}')
if all_rows:
    oldest_dt = datetime.datetime.utcfromtimestamp(min(all_rows)/1000)
    newest_dt = datetime.datetime.utcfromtimestamp(max(all_rows)/1000)
    days = (max(all_rows) - min(all_rows)) / (1000*3600*24)
    print(f'Range : {oldest_dt} -> {newest_dt}')
    print(f'Days  : {days:.1f}')
