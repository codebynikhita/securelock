import urllib.request
import re
import json
import sys
import socket
socket.setdefaulttimeout(8.0)

# ── Method 1: Instagram JSON API (real-time, exact counts) ─────────────────
def _fetch_json_api(username):
    """
    Uses Instagram's internal web_profile_info API.
    Returns exact real-time follower/following/post counts — not cached.
    No login required, just the right headers.
    """
    hosts = ['www.instagram.com', 'i.instagram.com']
    last_err = None
    data = None

    for host in hosts:
        url = f'https://{host}/api/v1/users/web_profile_info/?username={username}'
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/125.0.0.0 Safari/537.36'
            ),
            'x-ig-app-id': '936619743392459',
            'Accept': '*/*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Referer': 'https://www.instagram.com/',
            'Origin': 'https://www.instagram.com',
        }
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=6.0) as r:
                data = json.loads(r.read().decode('utf-8'))
                break
        except Exception as e:
            last_err = e

    if data is None:
        raise last_err if last_err else Exception("Failed to query Instagram JSON endpoints")

    user = data['data']['user']
    if not user:
        return None

    followers = user['edge_followed_by']['count']
    following  = user['edge_follow']['count']
    posts      = user['edge_owner_to_timeline_media']['count']
    pic_url    = user.get('profile_pic_url_hd') or user.get('profile_pic_url', '')
    is_private = user.get('is_private', False)
    full_name  = user.get('full_name', '')
    biography  = user.get('biography', '')

    # Estimate account age from user ID in pic URL or username lookup
    user_id_match = re.search(r'/v/[^/]+/(\d+)_', pic_url)
    owner_id = int(user_id_match.group(1)) if user_id_match else None

    return {
        'followers':       followers,
        'following':       following,
        'posts':           posts,
        'profile_pic_url': pic_url,
        'has_profile_pic': bool(pic_url) and 'anonymous' not in pic_url,
        'is_private':      is_private,
        'full_name':       full_name,
        'biography':       biography,
        'owner_id':        owner_id,
        'source':          'json_api',
    }


# ── Method 2: Googlebot HTML scrape (fallback, ~5% off due to caching) ─────
def _fetch_googlebot_html(username):
    """
    Fallback: Googlebot UA scrapes the public profile page.
    Numbers may be slightly stale (~5%) due to Instagram's bot-cache.
    """
    url = f'https://www.instagram.com/{username}/'
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
        'Connection': 'close'
    }

    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=8.0) as response:
        html = response.read().decode('utf-8')

    # Check 404 / not found
    if 'Page Not Found' in html or '"httpStatus":404' in html:
        return None

    def clean_num(s):
        s = s.replace(',', '').replace(' ', '').strip().upper()
        if not s: return 0
        if 'K' in s: return int(float(s.replace('K', '')) * 1000)
        if 'M' in s: return int(float(s.replace('M', '')) * 1000000)
        if 'B' in s: return int(float(s.replace('B', '')) * 1000000000)
        try: return int(float(s))
        except: return 0

    followers = following = posts = 0

    # Primary: og:description / meta tag
    meta = re.search(
        r'content=\"([0-9.,\sKMB]+)\s+Followers,\s+([0-9.,\sKMB]+)\s+Following,\s+([0-9.,\sKMB]+)\s+Posts',
        html, re.IGNORECASE
    )
    if meta:
        followers = clean_num(meta.group(1))
        following  = clean_num(meta.group(2))
        posts      = clean_num(meta.group(3))
    else:
        fol = re.search(r'([0-9.,KMB]+)\s*Followers', html, re.IGNORECASE)
        fng = re.search(r'([0-9.,KMB]+)\s*Following', html, re.IGNORECASE)
        pst = re.search(r'([0-9.,KMB]+)\s*Posts',     html, re.IGNORECASE)
        if fol: followers = clean_num(fol.group(1))
        if fng: following  = clean_num(fng.group(1))
        if pst: posts      = clean_num(pst.group(1))

    # Profile picture
    profile_pic_url = None
    og_img = re.search(r'property="og:image"\s+content="([^"]+)"', html)
    if not og_img:
        og_img = re.search(r'content="([^"]+)"\s+property="og:image"', html)
    if og_img:
        profile_pic_url = og_img.group(1)

    # Owner ID for age estimation
    owner_id = None
    om = re.search(r'instapp:owner_id"\s+content="(\d+)"', html)
    if om:
        owner_id = int(om.group(1))
    else:
        ids = re.findall(r'"owner_id":"(\d+)"', html)
        if ids: owner_id = int(ids[0])

    return {
        'followers':       followers,
        'following':       following,
        'posts':           posts,
        'profile_pic_url': profile_pic_url,
        'has_profile_pic': profile_pic_url is not None,
        'is_private':      False,
        'owner_id':        owner_id,
        'source':          'html_scrape',
    }


# ── Account age from owner_id ───────────────────────────────────────────────
def _estimate_age(owner_id):
    if not owner_id: return 3.5
    if owner_id > 100000000000: return 0.5
    if owner_id > 90000000000:  return 1.0
    if owner_id > 78000000000:  return 2.0
    if owner_id > 65000000000:  return 3.0
    if owner_id > 52000000000:  return 4.0
    if owner_id > 38000000000:  return 5.0
    if owner_id > 25000000000:  return 6.0
    if owner_id > 15000000000:  return 7.0
    if owner_id > 8000000000:   return 8.0
    if owner_id > 2000000000:   return 9.5
    if owner_id > 500000000:    return 12.0
    return 14.5


# ── Public entry point ──────────────────────────────────────────────────────
def get_instagram_metrics(username):
    username = username.lstrip('@')
    json_error = None

    # Try JSON API first (exact real-time counts)
    try:
        result = _fetch_json_api(username)
        if result is not None:
            result['username']    = username
            result['status']      = 'success'
            result['account_age'] = _estimate_age(result.get('owner_id'))
            return result
    except Exception as e:
        json_error = str(e)

    # Fallback: Googlebot HTML scrape
    try:
        result = _fetch_googlebot_html(username)
        if result is None:
            return {'status': 'error', 'message': 'Account not found'}
        result['username']    = username
        result['status']      = 'success'
        result['account_age'] = _estimate_age(result.get('owner_id'))
        result['json_api_error'] = json_error
        return result
    except Exception as e:
        return {'status': 'error', 'message': str(e)}


if __name__ == '__main__':
    target = sys.argv[1] if len(sys.argv) > 1 else 'nasa'
    print(json.dumps(get_instagram_metrics(target), indent=2))
