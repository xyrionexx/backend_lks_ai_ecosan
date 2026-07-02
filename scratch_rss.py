import urllib.request
import xml.etree.ElementTree as ET

url = "https://news.google.com/rss/search?q=sampah+lingkungan&hl=id&gl=ID&ceid=ID:id"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
try:
    with urllib.request.urlopen(req) as response:
        xml_data = response.read()
    root = ET.fromstring(xml_data)
    items = root.findall('.//item')
    print(f"Found {len(items)} news items.")
    for item in items[:3]:
        title = item.find('title').text
        link = item.find('link').text
        pubDate = item.find('pubDate').text
        print(f"- {title}\n  {link}\n")
except Exception as e:
    print("Error:", e)
