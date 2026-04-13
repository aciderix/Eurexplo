#!/usr/bin/env python3
"""Scrapeur parallel de turf-fr.com pour les cotes du jour."""
import requests
from bs4 import BeautifulSoup
import re, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml',
    'Accept-Language': 'fr-FR,fr;q=0.9',
}
BASE = 'https://www.turf-fr.com'
MAX_WORKERS = 10


def get_course_urls():
    r = requests.get(f'{BASE}/courses-pmu/partants', headers=HEADERS, timeout=15)
    soup = BeautifulSoup(r.text, 'html.parser')
    seen = set()
    urls = []
    for a in soup.find_all('a', href=re.compile(r'/courses-pmu/partants/r\d+-')):
        if a['href'] not in seen:
            seen.add(a['href'])
            urls.append(a['href'])
    return urls


def scrape_one(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(r.text, 'html.parser')
        
        meta = {
            'course_id': soup.find('input', {'id': 'course_id'})['value'] if soup.find('input', {'id': 'course_id'}) else None,
            'course_name': soup.find('input', {'id': 'tab_name_menu'})['value'] if soup.find('input', {'id': 'tab_name_menu'}) else None,
            'date': soup.find('input', {'id': 'selectedreuniondate'})['value'] if soup.find('input', {'id': 'selectedreuniondate'}) else None,
            'reunion': soup.find('input', {'id': 'selectedreunionnumero'})['value'] if soup.find('input', {'id': 'selectedreunionnumero'}) else None,
            'course_num': soup.find('input', {'id': 'selectedcourseindex'})['value'] if soup.find('input', {'id': 'selectedcourseindex'}) else None,
            'url': url,
        }
        
        runners = []
        for row in soup.find_all('tr', class_='ligne_partant'):
            cells = row.find_all('td')
            if len(cells) < 13:
                continue
            
            cote_attr = row.get('cote', '')
            cote = float(cote_attr) if cote_attr else None
            
            name_a = cells[1].find('a')
            name = name_a.get_text(strip=True) if name_a else cells[1].get_text(strip=True)
            
            if not name:
                continue
            
            runners.append({
                'numero': cells[0].get_text(strip=True),
                'nom': name,
                'cote': cote,
                'musique': cells[12].get_text(strip=True)[:30] if len(cells) > 12 else '',
                'sexe_age': cells[6].get_text(strip=True) if len(cells) > 6 else '',
                'jockey': cells[7].get_text(strip=True) if len(cells) > 7 else '',
                'entraineur': cells[8].get_text(strip=True) if len(cells) > 8 else '',
                'record': cells[10].get_text(strip=True) if len(cells) > 10 else '',
                'gains': cells[11].get_text(strip=True) if len(cells) > 11 else '',
            })
        
        return {'meta': meta, 'runners': runners, 'url': url}
    except Exception as e:
        return {'meta': {'url': url}, 'runners': [], 'error': str(e)}


def scrape_all():
    print('=== Scrapeur turf-fr.com (parallel) ===')
    urls = get_course_urls()
    print(f'{len(urls)} courses trouvées')
    
    all_courses = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(scrape_one, url): url for url in urls}
        done = 0
        for future in as_completed(futures):
            done += 1
            result = future.result()
            if result['runners']:
                all_courses.append(result)
                wc = sum(1 for r in result['runners'] if r['cote'])
                print(f'  [{done}/{len(urls)}] {result["meta"].get("course_name","?")[:50]} | {len(result["runners"])} partants, {wc} cotes')
    
    return all_courses


if __name__ == '__main__':
    t0 = time.time()
    courses = scrape_all()
    print(f'\n=== {len(courses)} courses en {time.time()-t0:.1f}s ===')
    
    # Trier par reunion/course
    courses.sort(key=lambda x: (x['meta'].get('reunion',''), x['meta'].get('course_num','')))
    
    for c in courses:
        m = c['meta']
        runners = c['runners']
        with_cotes = sorted([r for r in runners if r['cote']], key=lambda x: x['cote'])
        if with_cotes:
            print(f"\n  [{m.get('reunion')}-C{m.get('course_num')}] {m.get('course_name','')[:60]} | {len(with_cotes)}/{len(runners)} cotes")
            for r in with_cotes[:5]:
                print(f"    #{r['numero']:>2} {r['nom'][:25]:25s} {r['cote']:>5.1f}/1  {r.get('musique','')[:15]}")
    
    out = '/tmp/turffr_cotes_today.json'
    with open(out, 'w') as f:
        json.dump(courses, f, ensure_ascii=False, indent=2)
    print(f'\nSauvegardé: {out}')
