import urllib.request
import re
import json
import sys
import socket
socket.setdefaulttimeout(5.0)

def get_instagram_metrics(username):
    username = username.lstrip('@')
    url = f"https://www.instagram.com/{username}/"
    headers = {
        'User-Agent': 'Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)',
        'Connection': 'close'
    }
    
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=5.0) as response:
            html = response.read().decode('utf-8')
            
        # Parse followers, following, posts from meta tag:
        # e.g., "273 Followers, 234 Following, 33 Posts - ..."
        meta_pattern = r'content="([0-9.,\sKMB]+)\s+Followers,\s+([0-9.,\sKMB]+)\s+Following,\s+([0-9.,\sKMB]+)\s+Posts'
        match = re.search(meta_pattern, html, re.IGNORECASE)
        
        followers = 0
        following = 0
        posts = 0
        
        def clean_num(s):
            s = s.replace(',', '').replace(' ', '').strip().upper()
            if not s: return 0
            if 'K' in s:
                return int(float(s.replace('K', '')) * 1000)
            if 'M' in s:
                return int(float(s.replace('M', '')) * 1000000)
            try:
                return int(float(s))
            except:
                return 0
                
        if match:
            followers = clean_num(match.group(1))
            following = clean_num(match.group(2))
            posts = clean_num(match.group(3))
        else:
            # Fallback patterns without nested quantifiers
            fol_match = re.search(r'([0-9.,KMB]+)\s*(?:followers|Followers)', html)
            fng_match = re.search(r'([0-9.,KMB]+)\s*(?:following|Following)', html)
            pst_match = re.search(r'([0-9.,KMB]+)\s*(?:posts|Posts|posts_count)', html)
            
            if fol_match: followers = clean_num(fol_match.group(1))
            if fng_match: following = clean_num(fng_match.group(1))
            if pst_match: posts = clean_num(pst_match.group(1))
            
        # Extract Profile Picture URL
        profile_pic_url = None
        # Try og:image meta tag (most reliable)
        og_img = re.search(r'property="og:image"\s+content="([^"]+)"', html)
        if not og_img:
            og_img = re.search(r'content="([^"]+)"\s+property="og:image"', html)
        if og_img:
            profile_pic_url = og_img.group(1)
        # Fallback: look for profile_pic_url in embedded JSON
        if not profile_pic_url:
            pic_match = re.search(r'"profile_pic_url":"([^"]+)"', html)
            if pic_match:
                profile_pic_url = pic_match.group(1).replace('\\u0026', '&')

        # Extract Owner ID / User ID
        owner_id = None
        owner_match = re.search(r'instapp:owner_id"\s+content="(\d+)"', html)
        if owner_match:
            owner_id = int(owner_match.group(1))
        else:
            owner_match2 = re.search(r'instagram://user\?username=[\w.]*\&amp;id=(\d+)', html)
            if owner_match2:
                owner_id = int(owner_match2.group(1))
            else:
                ids = re.findall(r'"owner_id":"(\d+)"', html)
                if ids:
                    owner_id = int(ids[0])
                else:
                    ids2 = re.findall(r'"id":"(\d+)"', html)
                    if ids2:
                        for oid in ids2:
                            if 5 < len(oid) < 15:
                                owner_id = int(oid)
                                break
                                
        # Calculate estimated account age based on owner ID
        account_age = 3.5
        if owner_id:
            if owner_id > 100000000000:    account_age = 0.5
            elif owner_id > 90000000000:   account_age = 1.0
            elif owner_id > 78000000000:   account_age = 2.0
            elif owner_id > 65000000000:   account_age = 3.0
            elif owner_id > 52000000000:   account_age = 4.0
            elif owner_id > 38000000000:   account_age = 5.0
            elif owner_id > 25000000000:   account_age = 6.0
            elif owner_id > 15000000000:   account_age = 7.0
            elif owner_id > 8000000000:    account_age = 8.0
            elif owner_id > 2000000000:    account_age = 9.5
            elif owner_id > 500000000:     account_age = 12.0
            else:                          account_age = 14.5
            
        return {
            'status': 'success',
            'username': username,
            'followers': followers,
            'following': following,
            'posts': posts,
            'account_age': account_age,
            'owner_id': owner_id,
            'profile_pic_url': profile_pic_url,
            'has_profile_pic': profile_pic_url is not None
        }
    except Exception as e:
        return {
            'status': 'error',
            'message': str(e)
        }

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print(json.dumps(get_instagram_metrics(sys.argv[1]), indent=2))
    else:
        print(json.dumps(get_instagram_metrics('__nikhita__09'), indent=2))
