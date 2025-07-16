import os
import time
import requests
from bs4 import BeautifulSoup

HEADERS = {'User-Agent': 'Mozilla/5.0'}

COMPANIES = {
    'GOOGL': '1652044',
    'MSFT': '789019',
    'NVDA': '1045810'
}

TARGET_YEARS = ['2022', '2023', '2024']
BASE_DIR = "data"

def get_10k_filing_links(cik):
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=10-K&owner=exclude&count=100"
    res = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(res.text, 'html.parser')
    
    links = []
    rows = soup.find_all('tr')
    for row in rows:
        cells = row.find_all('td')
        if len(cells) >= 4 and '10-K' in cells[0].text:
            filing_date = cells[3].text.strip()
            year = filing_date.split('-')[0]
            if year in TARGET_YEARS:
                doc_page_tag = row.find('a', href=True)
                if doc_page_tag:
                    doc_page_url = "https://www.sec.gov" + doc_page_tag['href']
                    links.append((year, doc_page_url))
    return links

def get_htm_link(doc_page_url):
    res = requests.get(doc_page_url, headers=HEADERS)
    soup = BeautifulSoup(res.text, 'html.parser')
    table = soup.find('table', class_='tableFile')

    if not table:
        return None

    for row in table.find_all('tr')[1:]:
        cells = row.find_all('td')
        if len(cells) < 4:
            continue
        doc_type = cells[3].text.strip().lower()
        doc_name = cells[1].text.strip().lower()
        if '10-k' in doc_type or '10-k' in doc_name:
            href = cells[2].find('a', href=True)['href']
            return "https://www.sec.gov" + href
    return None

def download_file(url, save_path):
    try:
        res = requests.get(url, headers=HEADERS)
        if res.status_code == 200:
            with open(save_path, 'wb') as f:
                f.write(res.content)
            print(f"[✓] Saved: {save_path}")
        else:
            print(f"[!] Failed to download {url} (Status code: {res.status_code})")
    except Exception as e:
        print(f"[X] Error: {e}")

def main():
    os.makedirs(BASE_DIR, exist_ok=True)

    for company, cik in COMPANIES.items():
        print(f"\n🔍 Looking for 10-K filings for {company}")
        links = get_10k_filing_links(cik)
        for year, filing_url in links:
            filename = f"{company}_{year}.htm"
            filepath = os.path.join(BASE_DIR, filename)
            if os.path.exists(filepath):
                print(f"[!] Already exists: {filename}")
                continue
            print(f"→ Downloading {company} {year}")
            htm_link = get_htm_link(filing_url)
            if htm_link:
                download_file(htm_link, filepath)
                time.sleep(1)
            else:
                print(f"[!] No .htm link found for {company} {year}")

if __name__ == "__main__":
    main()
