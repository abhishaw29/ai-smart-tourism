import os
import json
import math
import re
import io
import html
import hashlib
import time
import base64
import difflib
import textwrap
from pathlib import Path
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import numpy as np
import datetime
import pandas as pd
import requests
import streamlit as st
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from urllib.parse import quote_plus, quote
import functools
import calendar
import hmac
import secrets

# ---- Accounts, calendar and packing helpers (previously trip_tools.py) ----
IST = datetime.timezone(datetime.timedelta(hours=5, minutes=30))
USERS_PATH = Path("user_data") / "users.json"
AVATARS = ["🧭", "🎒", "🌴", "🏔️", "🛕", "🚂", "📷", "🦚"]
TRAVEL_STYLES = ["Relaxed", "Adventure", "Heritage & culture", "Spiritual", "Nature & wildlife", "Food & markets", "Family", "Budget backpacking"]


# ----------------------------------------------------------------------------
# Time / greeting
# ----------------------------------------------------------------------------
def now_ist():
    return datetime.datetime.now(IST)


def greeting_for(hour):
    if hour < 5:
        return "Good night"
    if hour < 12:
        return "Good morning"
    if hour < 17:
        return "Good afternoon"
    if hour < 21:
        return "Good evening"
    return "Good night"


# ----------------------------------------------------------------------------
# Accounts (local JSON store)
# ----------------------------------------------------------------------------
def _read_users():
    try:
        data = json.loads(USERS_PATH.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _write_users(users):
    USERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = USERS_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(users, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(USERS_PATH)


def _hash(password, salt_hex):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), 200_000).hex()


def default_profile(name="Traveller"):
    return {
        "name": name, "avatar": AVATARS[0], "home_city": "", "styles": [],
        "favorites": [], "saved_trips": [], "custom_days": [], "packing": {},
    }


def create_user(username, password, name, legacy_favorites=None):
    """Returns (ok, message)."""
    username = (username or "").strip().lower()
    if not re.fullmatch(r"[a-z0-9_.]{3,24}", username):
        return False, "Username must be 3-24 characters: letters, numbers, dot or underscore."
    if len(password or "") < 6:
        return False, "Password must be at least 6 characters."
    users = _read_users()
    if username in users:
        return False, "That username is already taken."
    salt = secrets.token_hex(16)
    prof = default_profile((name or "").strip() or username.title())
    if legacy_favorites:
        prof["favorites"] = [str(x) for x in legacy_favorites]
    users[username] = {"salt": salt, "hash": _hash(password, salt), "profile": prof}
    _write_users(users)
    return True, "Account created."


def verify_user(username, password):
    """Returns the profile dict on success, otherwise None."""
    rec = _read_users().get((username or "").strip().lower())
    if not rec:
        return None
    if hmac.compare_digest(rec["hash"], _hash(password or "", rec["salt"])):
        prof = default_profile()
        prof.update(rec.get("profile", {}))
        return prof
    return None


def save_profile(username, profile):
    users = _read_users()
    key = (username or "").strip().lower()
    if key in users:
        users[key]["profile"] = profile
        _write_users(users)


# ----------------------------------------------------------------------------
# Calendar
# ----------------------------------------------------------------------------
def month_events(year, month, holiday_fn, custom_days=None):
    """[(date, label, kind)] for the month. kind: holiday | custom"""
    out = []
    for d in range(1, calendar.monthrange(year, month)[1] + 1):
        date = datetime.date(year, month, d)
        for name, _kind in holiday_fn(date):
            out.append((date, name, "holiday"))
    for item in custom_days or []:
        try:
            cd = datetime.date.fromisoformat(item["date"])
        except Exception:
            continue
        if cd.year == year and cd.month == month:
            out.append((cd, item.get("label", "My day"), "custom"))
    return sorted(out, key=lambda x: x[0])


def long_weekends(year, month, holiday_fn):
    """Find 3+ day breaks around holidays, and 'bridge' ideas (Tue/Thu holidays)."""
    def off(d):
        return d.weekday() >= 5 or bool(holiday_fn(d))

    found, seen = [], set()
    for date, name, kind in month_events(year, month, holiday_fn):
        if kind != "holiday" or date in seen:
            continue
        a = b = date
        while off(a - datetime.timedelta(days=1)):
            a -= datetime.timedelta(days=1)
        while off(b + datetime.timedelta(days=1)):
            b += datetime.timedelta(days=1)
        length = (b - a).days + 1
        if length >= 3:
            found.append({"start": a, "end": b, "days": length, "why": name, "tip": "No leave needed."})
            seen.update(a + datetime.timedelta(days=i) for i in range(length))
        elif date.weekday() in (1, 3):  # Tue / Thu
            gap = date - datetime.timedelta(days=1) if date.weekday() == 1 else date + datetime.timedelta(days=1)
            s, e = sorted([gap, date])
            ext_s, ext_e = s, e
            while off(ext_s - datetime.timedelta(days=1)):
                ext_s -= datetime.timedelta(days=1)
            while off(ext_e + datetime.timedelta(days=1)):
                ext_e += datetime.timedelta(days=1)
            found.append({"start": ext_s, "end": ext_e, "days": (ext_e - ext_s).days + 1, "why": name,
                          "tip": f"Take 1 day off ({gap.strftime('%a %d %b')}) for a {(ext_e - ext_s).days + 1}-day break."})
            seen.add(date)
    return found


def month_calendar_html(year, month, holiday_fn, trip_start=None, trip_end=None, today=None, custom_days=None):
    """A responsive month grid. Holidays get a dot (name shown on wide screens)."""
    today = today or datetime.date.today()
    cal = calendar.Calendar(firstweekday=6)  # Sunday first
    hol, cust = {}, {}
    for date, label, kind in month_events(year, month, holiday_fn, custom_days):
        (hol if kind == "holiday" else cust).setdefault(date, []).append(label)
    heads = "".join(f'<div class="tc-h">{d}</div>' for d in ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"])
    cells = []
    for week in cal.monthdatescalendar(year, month):
        for d in week:
            cls = ["tc-d"]
            if d.month != month:
                cls.append("tc-out")
            if d.weekday() >= 5:
                cls.append("tc-we")
            if d == today:
                cls.append("tc-today")
            if trip_start and trip_end and trip_start <= d <= trip_end:
                cls.append("tc-trip")
                if d == trip_start:
                    cls.append("tc-tstart")
                if d == trip_end:
                    cls.append("tc-tend")
            label_html, title = "", []
            if d.month == month and d in hol:
                cls.append("tc-hol")
                label_html += f'<span class="tc-l">{html.escape(hol[d][0])}</span>'
                title += hol[d]
            if d.month == month and d in cust:
                cls.append("tc-cus")
                label_html += f'<span class="tc-l tc-lc">★ {html.escape(cust[d][0])}</span>'
                title += cust[d]
            t = f' title="{html.escape(", ".join(title))}"' if title else ""
            cells.append(f'<div class="{" ".join(cls)}"{t}><b>{d.day}</b>{label_html}</div>')
    return (f'<div class="tc-wrap"><div class="tc-grid">{heads}{"".join(cells)}</div>'
            '<div class="tc-legend"><span><i class="lg lg-today"></i>Today</span>'
            '<span><i class="lg lg-hol"></i>Holiday / festival</span>'
            '<span><i class="lg lg-trip"></i>Your trip</span>'
            '<span><i class="lg lg-cus"></i>My special day</span>'
            '<span><i class="lg lg-we"></i>Weekend</span></div></div>')


CALENDAR_CSS = """
<style>
.tc-wrap{background:#fff;border:1px solid #DCE3D1;border-radius:16px;padding:12px;box-shadow:0 4px 14px rgba(53,69,34,.07)}
.tc-grid{display:grid;grid-template-columns:repeat(7,minmax(0,1fr));gap:4px}
.tc-h{text-align:center;font-size:12px;font-weight:700;color:#59634D;padding:4px 0}
.tc-d{position:relative;min-height:74px;border:1px solid #E8EDDF;border-radius:10px;padding:6px 6px 4px;background:#FBFCF8;overflow:hidden}
.tc-d b{font-size:14px;color:#27311F}
.tc-out{opacity:.35}
.tc-we{background:#F3F5EC}
.tc-hol{background:#FBEBE3;border-color:#E9B9A3}
.tc-hol b{color:#9E4A2A}
.tc-cus{box-shadow:inset 0 -3px 0 #71834A}
.tc-today{outline:2px solid #354522;outline-offset:-2px}
.tc-trip{background:#E3EBD2;border-color:#71834A}
.tc-tstart,.tc-tend{background:#71834A}
.tc-tstart b,.tc-tend b{color:#fff}
.tc-l{display:block;font-size:10.5px;line-height:1.25;color:#9E4A2A;margin-top:3px;font-weight:600}
.tc-lc{color:#354522}
.tc-legend{display:flex;flex-wrap:wrap;gap:6px 14px;margin-top:10px;font-size:12px;color:#59634D}
.tc-legend .lg{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:5px;vertical-align:-1px;border:1px solid #C9D3BC}
.lg-today{outline:2px solid #354522;background:#fff}.lg-hol{background:#FBEBE3}.lg-trip{background:#71834A}
.lg-cus{background:#fff;box-shadow:inset 0 -3px 0 #71834A}.lg-we{background:#F3F5EC}
@media(max-width:640px){
 .tc-d{min-height:48px;padding:4px 2px;text-align:center}
 .tc-d b{font-size:13px}
 .tc-l{display:none}
 .tc-hol::after{content:"";position:absolute;left:50%;bottom:5px;width:6px;height:6px;margin-left:-3px;border-radius:50%;background:#C2603A}
 .tc-h{font-size:11px}
}
</style>
"""


# ----------------------------------------------------------------------------
# Packing list
# ----------------------------------------------------------------------------
def season_of(month):
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "summer"
    if month in (6, 7, 8, 9):
        return "monsoon"
    return "autumn"


PLACE_KEYS = {
    "beach": ["beach", "island", "coast", "sea"],
    "temple": ["temple", "church", "mosque", "gurudwara", "shrine", "ashram", "monastery", "religious", "pilgrim", "spiritual", "cathedral"],
    "hills": ["hill", "mountain", "trek", "valley", "peak", "snow", "glacier"],
    "wildlife": ["wildlife", "sanctuary", "national park", "safari", "tiger", "reserve", "forest"],
    "desert": ["desert", "dune"],
    "water": ["waterfall", "lake", "river", "backwater"],
    "heritage": ["fort", "palace", "monument", "museum", "tomb", "memorial", "heritage", "temple complex", "cave"],
}
ALTITUDE_STATES = {"ladakh", "himachal pradesh", "uttarakhand", "sikkim", "jammu and kashmir", "jammu & kashmir", "arunachal pradesh"}
PERMIT_STATES = {"arunachal pradesh", "nagaland", "mizoram", "manipur", "lakshadweep"}


def place_kinds(types):
    text = " ".join(str(t).lower() for t in types)
    return {k for k, words in PLACE_KEYS.items()
            if any(re.search(r"\b" + re.escape(w) + r"s?\b", text) for w in words)}


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:48]


def build_packing_list(month, days, kinds, transport="Train", extras=(), states=(), no_dslr=()):
    """Returns an ordered dict {category: [(item_id, label), ...]}."""
    days = max(int(days or 1), 1)
    season = season_of(int(month))
    states = {str(s).lower() for s in states}
    out = {}

    def add(cat, *labels):
        bucket = out.setdefault(cat, [])
        for lb in labels:
            if lb and all(lb != x[1] for x in bucket):
                bucket.append((slug(cat + "-" + lb), lb))

    add("📄 Documents & money",
        "Photo ID (Aadhaar / Passport / Voter ID / Driving licence)",
        "Tickets and hotel bookings (screenshots saved offline)",
        "Some cash in small notes + UPI / cards",
        "Emergency contacts written on paper")
    if transport == "Car":
        add("📄 Documents & money", "Driving licence, RC and insurance papers", "FASTag balance checked")
    if transport == "Flight":
        add("📄 Documents & money", "Web check-in done / boarding pass saved", "Baggage allowance checked with the airline")
    if states & PERMIT_STATES:
        add("📄 Documents & money", "Inner Line Permit / entry permit (check the official portal before you go)")

    add("👕 Clothing",
        f"Outfits for the trip (about {min(days, 5) + 1} sets; re-wear to travel light)",
        f"Underwear and socks ({days + 1} pairs)", "Comfortable walking shoes", "Light sleepwear")
    if season == "summer":
        add("👕 Clothing", "Light cotton clothes", "Cap or hat", "Sunglasses")
    if season == "monsoon":
        add("👕 Clothing", "Raincoat or compact umbrella", "Quick-dry clothes", "Spare footwear (shoes get wet)", "Waterproof cover for your bag")
    if season == "winter":
        add("👕 Clothing", "Sweater or jacket", "Muffler / cap", "Thermal innerwear for cold nights")
    if season == "autumn":
        add("👕 Clothing", "One light jacket for cool mornings and evenings")

    add("🧴 Health & toiletries",
        "Your regular medicines + prescriptions", "Basic first-aid (bandages, antiseptic, pain relief, ORS)",
        "Hand sanitiser and wet wipes", "Toothbrush, soap, shampoo, other toiletries", "Reusable water bottle")
    if season in ("summer", "autumn", "winter"):
        add("🧴 Health & toiletries", "Sunscreen and lip balm")
    if season == "monsoon" or "wildlife" in kinds or "water" in kinds:
        add("🧴 Health & toiletries", "Mosquito repellent")

    add("🔌 Electronics", "Phone + charger", "Power bank (cabin bag only if flying)", "Offline maps downloaded for the destination", "Earphones")
    if no_dslr:
        add("🔌 Electronics", "Phone camera ready (DSLR not allowed at: " + ", ".join(sorted(no_dslr)[:4]) + ")")

    if "beach" in kinds:
        add("🏖️ Beach & water", "Swimwear", "Quick-dry towel", "Flip-flops", "Waterproof phone pouch", "Sunscreen (water-resistant)")
    if "temple" in kinds:
        add("🛕 Temples & worship places", "Modest clothes covering shoulders and knees", "Socks (stone floors get very hot)",
            "Slip-on footwear (easy to leave outside)", "Scarf / dupatta for head covering", "Small bag (many shrines restrict large bags)")
    if "hills" in kinds or states & ALTITUDE_STATES:
        add("⛰️ Hills & mountains", "Layers: fleece + windcheater", "Gloves and woollen cap", "Shoes with good grip",
            "Go slow on arrival and drink plenty of water (ask your doctor about altitude medicine)")
    if "wildlife" in kinds:
        add("🐯 Wildlife & safari", "Neutral colours (khaki / green / brown)", "Binoculars", "Closed shoes and full-sleeve top", "ID proof for park entry")
    if "desert" in kinds:
        add("🏜️ Desert", "Cotton scarf for dust and sun", "Moisturiser and lip balm", "Sunglasses")
    if "water" in kinds:
        add("💧 Lakes & waterfalls", "Slip-resistant sandals", "Dry bag for phone and wallet", "Change of clothes")
    if "heritage" in kinds:
        add("🏰 Forts & monuments", "Comfortable shoes (lots of steps)", "Hat and water for open courtyards")

    if "Kids" in extras:
        add("👶 Kids", "Snacks and a water bottle", "Spare clothes (2 sets)", "Kids' medicines and thermometer", "Favourite toy or book")
    if "Senior citizens" in extras:
        add("👴 Senior travellers", "Regular medicines for the whole trip + spare", "Doctor's prescription copy", "Walking stick / knee support if used", "Spare glasses")
    if "Photography" in extras:
        add("📷 Photography", "Extra batteries and memory cards", "Lens cloth", "Tripod (check if allowed at each place)")
    if "Pilgrimage" in extras:
        add("🙏 Pilgrimage", "Offerings / puja items if needed", "Clean change of clothes", "Water + light snacks for long queues")

    if transport == "Train":
        add("🚆 Journey", "Light blanket or shawl", "Snacks for the ride", "Earplugs / eye mask")
    elif transport == "Bus":
        add("🚌 Journey", "Neck pillow", "Motion-sickness tablets if you need them", "Snacks and water")
    elif transport == "Car":
        add("🚗 Journey", "Spare tyre + jack checked", "Car charger and phone holder", "Snacks and water")
    elif transport == "Flight":
        add("✈️ Journey", "Valuables and meds in cabin bag", "Padlock for checked bag")
    return out

# ---- end helpers ----

APP_VERSION = "2.4.0"

# =====================================================
# PAGE CONFIGURATION
# =====================================================
st.set_page_config(
    page_title="AI Smart Tourist Planner",
    page_icon="🌍",
    layout="wide"
)
BASE_DIR = Path(__file__).resolve().parent
DATA_PATHS = [
    BASE_DIR / "dataset" / "cleaned_tourist_dataset_v2.csv",
    BASE_DIR / "cleaned_tourist_dataset_v2.csv",
    BASE_DIR / "dataset" / "Top Indian Places to Visit.csv",
    BASE_DIR / "Top_Indian_Places_to_Visit_Updated.csv",
    BASE_DIR / "Top Indian Places to Visit.csv",
]
# =====================================================
# CATEGORIES & PRESETS
# =====================================================
CATEGORIES = {
    "Beaches": ["beach", "beaches", "sea", "coast", "coastal", "ocean", "shore", "seaside"],
    "Lakes & Rivers": ["lake", "lakes", "reservoir", "wetland", "river", "waterbody", "backwater"],
    "Waterfalls": ["waterfall", "waterfalls", "fall", "falls", "cascade"],
    "Temples & Religious Places": ["temple", "church", "mosque", "gurudwara", "gurudwara", "spiritual", "religious", "pilgrimage", "ashram", "monastery", "shrine"],
    "Mountains & Hill Stations": ["mountain", "mountains", "hill", "hills", "hill station", "peak", "summit", "valley", "trek", "hiking"],
    "Historical & Heritage": ["fort", "palace", "history", "heritage", "monument", "museum", "historic", "archaeological"],
    "Nature, Parks & Gardens": ["park", "wildlife", "sanctuary", "nature", "garden", "forest", "national park", "botanical"],
    "Adventure & Safari": ["adventure", "trek", "safari", "rafting", "camping", "hiking", "zipline"],
    "Islands": ["island", "islands", "archipelago"],
    "Deserts": ["desert", "sand dunes", "dune"]
}

BUDGET_PRESETS = {
    "Budget": {"hotel": 1000, "food": 300, "local": 150},
    "Mid-range": {"hotel": 2500, "food": 600, "local": 300},
    "Luxury": {"hotel": 7000, "food": 1500, "local": 800},
}

TRANSPORT_RATES = {
    "Train (approximate)": 1.5,
    "Bus (road-distance estimate)": 2.5,
    "Car": 12.0,
    "Flight (approximate)": 5.0,
}
PER_VEHICLE_MODES = {"Car"}  # cost is shared by everyone in the vehicle

FALLBACK_IMAGES = [
    "https://images.unsplash.com/photo-1506461883276-594a12b11cf3?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1532375810709-75b1da00537c?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1596176530529-78163a4f7af2?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1507525428034-b723cf961d3e?auto=format&fit=crop&w=800&q=80",
    "https://images.unsplash.com/photo-1519904981063-b0cf448d479e?auto=format&fit=crop&w=800&q=80"
]

FAV_PATH = Path("user_data") / "favorites.json"


def load_favorites():
    try:
        data = json.loads(FAV_PATH.read_text(encoding="utf-8"))
        return [str(x) for x in data] if isinstance(data, list) else []
    except Exception:
        return []


def save_favorites():
    """Favourites are stored in the logged-in user's profile (guests keep them for the session only)."""
    user, prof = st.session_state.get("user"), st.session_state.get("profile")
    if prof is not None and user and not user.get("guest"):
        prof["favorites"] = list(st.session_state["favorites"])
        save_profile(user["username"], prof)


def clear_favorites():
    st.session_state["favorites"] = []
    save_favorites()


def set_global_state(name):
    st.session_state["global_state"] = name


def keep(key, default, options=None, multi=False):
    """Seed a widget key so its value survives page switches; drops values that are no longer valid options."""
    cur = st.session_state.get(key, default)
    if options is not None:
        opts = list(options)
        if multi:
            cur_list = list(cur) if isinstance(cur, (list, tuple)) else [cur]
            cur = [v for v in cur_list if v in opts]
            if not cur and key not in st.session_state:
                cur = [v for v in default if v in opts]
        elif cur not in opts:
            cur = default if default in opts else (opts[0] if opts else None)
    if cur is not None:
        st.session_state[key] = cur


def keep_range(key, default, lo, hi):
    cur = st.session_state.get(key, default)
    try:
        cur = min(max(cur, lo), hi)
    except TypeError:
        cur = default
    st.session_state[key] = cur


if "favorites" not in st.session_state:
    st.session_state["favorites"] = []  # filled from the profile after login

# =====================================================
# API CALLS & MACHINE LEARNING HELPERS
# =====================================================
@st.cache_data(ttl=86400, show_spinner=False)
def geocode_destination(name, city="", state=""):
    query = ", ".join(x for x in [str(name).strip(), str(city).strip(), str(state).strip(), "India"] if x)
    try:
        time.sleep(0.5)
        response = requests.get(
            "https://nominatim.openstreetmap.org/search",
            params={"q": query, "format": "jsonv2", "limit": 1, "countrycodes": "in"},
            headers=OSM_HEADERS,
            timeout=8
        )
        if response.status_code == 200:
            results = response.json()
            if results:
                lat = float(results[0]["lat"])
                lon = float(results[0]["lon"])
                if -90 <= lat <= 90 and -180 <= lon <= 180:
                    return (lat, lon)
    except Exception:
        pass
    return None

def render_interactive_route_map(route_df):
    valid_df = route_df.dropna(subset=['_latitude', '_longitude']).copy()
    if valid_df.empty:
        st.warning("No valid GPS coordinates available for mapping this itinerary.")
        return

    center_lat = valid_df['_latitude'].mean()
    center_lon = valid_df['_longitude'].mean()

    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")
    color_map = {
        "historical": "red", "monument": "darkred", "religious": "blue",
        "temple": "purple", "park": "green", "nature": "green",
        "museum": "orange", "shopping": "cadetblue", "beach": "lightblue"
    }

    route_coords = []
    for idx, row in valid_df.reset_index(drop=True).iterrows():
        lat, lon = row['_latitude'], row['_longitude']
        route_coords.append((lat, lon))
        place_type = str(row.get('_type', '')).lower()
        marker_color = "gray"
        for key, color in color_map.items():
            if key in place_type:
                marker_color = color
                break

        popup_html = f"""
        <div style="font-family: Arial, sans-serif; width: 180px;">
            <h4 style="margin: 0; color: #1E3A8A;">Stop {idx + 1}: {row['_name']}</h4>
            <p style="margin: 4px 0;"><b>City:</b> {row.get('_city', 'N/A')}</p>
            <p style="margin: 4px 0;"><b>Category:</b> {row.get('_type', 'Attraction')}</p>
        </div>
        """
        folium.Marker(
            location=[lat, lon],
            popup=folium.Popup(popup_html, max_width=220),
            tooltip=f"Stop {idx + 1}: {row['_name']}",
            icon=folium.Icon(color=marker_color, icon="info-sign", prefix="fa")
        ).add_to(m)

    if len(route_coords) > 1:
        folium.PolyLine(
            locations=route_coords, color="#2563EB", weight=4, opacity=0.85, dash_array="6, 6"
        ).add_to(m)

    st_folium(m, width="100%", height=500, returned_objects=[])

@st.cache_data(ttl=1800, show_spinner=False)
def get_weather(lat, lon):
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,apparent_temperature",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
                "timezone": "auto", "forecast_days": 3
            },
            timeout=8
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None

@st.cache_resource
def build_tfidf(texts):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix

def haversine_km(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
MONTH_ABBR = [m[:3].lower() for m in MONTHS]
BUDGET_FEE_CAP = {"Budget": 100, "Mid-range": 500, "Luxury": 2000}
REACH_KM = {"Nearby (under 300 km)": 300, "Regional (under 800 km)": 800, "Anywhere in India": None}

TRAVELER_PROFILES = {
    "Solo": {"kw": ["adventure", "trek", "hostel", "heritage", "culture"], "easy": False},
    "Couple": {"kw": ["scenic", "sunset", "lake", "beach", "palace", "garden"], "easy": False},
    "Family with kids": {"kw": ["park", "zoo", "garden", "museum", "wildlife", "beach"], "easy": True},
    "Friends group": {"kw": ["adventure", "beach", "trek", "fort", "waterfall"], "easy": False},
    "Senior citizens": {"kw": ["temple", "garden", "museum", "lake", "spiritual"], "easy": True},
}
STRENUOUS_WORDS = ["trek", "peak", "climb", "hike", "mountain", "summit", "trail"]
OUTDOOR_WORDS = ["beach", "trek", "park", "garden", "waterfall", "wildlife", "sanctuary", "hill", "lake", "fort", "valley", "safari", "island", "peak"]

def season_fit(text, month_idx):
    t = str(text).lower()
    if re.search(r"all year|year[- ]round|throughout|any time|^\s*all\s*$", t):
        return 1.0  # "All" in the v2 dataset means good all year
    found = [MONTH_ABBR.index(m) for m in re.findall(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", t)]
    if not found:
        return 0.5
    if len(found) >= 2 and re.search(r"\bto\b|-|–|till|through", t):
        start, end = found[0], found[-1]
        span = [(start + i) % 12 for i in range((end - start) % 12 + 1)]
        return 1.0 if month_idx in span else 0.0
    return 1.0 if month_idx in found else 0.0

# Season rules by place type. Used only when the dataset gives no month information.
# Month numbers: 0 = January ... 11 = December.
SEASON_RULES = [
    (("ski",), {11, 0, 1, 2}, "winter-sports season", "Dec to Mar"),
    (("national park", "sanctuary", "tiger", "wildlife", "safari"), {10, 11, 0, 1, 2, 3}, "wildlife viewing season", "Nov to Apr"),
    (("waterfall",), {6, 7, 8, 9}, "waterfall season", "Jul to Oct"),
    (("beach", "island"), {9, 10, 11, 0, 1, 2}, "beach season", "Oct to Mar"),
    (("hill", "valley"), {2, 3, 4, 5, 8, 9}, "hill-station weather", "Mar to Jun and Sep to Oct"),
    (("fort", "palace", "monument", "tomb", "mausoleum", "stepwell", "garden", "park", "lake",
      "ghat", "cave", "memorial", "viewpoint"), {9, 10, 11, 0, 1, 2}, "comfortable sightseeing weather", "Oct to Mar"),
]
ALL_YEAR_PATTERN = r"all year|year[- ]round|throughout|any time|^\s*all\s*$"


def type_season_rule(type_text):
    t = str(type_text).lower()
    for words, months, label, span in SEASON_RULES:
        if any(w in t for w in words):
            return months, label, span
    return None


def season_fit_place(best_text, type_text, month_idx):
    """Season score: month data from the dataset wins; otherwise use the rule for the place type."""
    score = season_fit(best_text, month_idx)
    if score != 0.5:
        return score
    rule = type_season_rule(type_text)
    if rule is None:
        return 0.5
    return 1.0 if month_idx in rule[0] else 0.25


def season_scores(frame, month_idx):
    return np.array([season_fit_place(b, t, month_idx) for b, t in zip(frame["_best_time"], frame["_type"])], dtype=float)


def season_reason(best_text, type_text, month_idx, month_name):
    t = str(best_text).strip().lower()
    score = season_fit(best_text, month_idx)
    if score == 1.0 and re.search(ALL_YEAR_PATTERN, t):
        return "The dataset lists it as good all year."
    if score == 1.0:
        return f"The dataset lists {best_text} as the best time, which includes {month_name}."
    if score == 0.0:
        return f"The dataset lists {best_text} as the best time, so {month_name} is outside it."
    rule = type_season_rule(type_text)
    if rule is None:
        return "No month information is available for this type of place."
    months, label, span = rule
    if month_idx in months:
        return f"In season for {month_name} ({label}, {span})."
    return f"Off-season for {month_name}: {label} is {span}."


def budget_fit(fee, tier):
    if pd.isna(fee): return 0.7
    cap = BUDGET_FEE_CAP[tier]
    return 1.0 if fee <= cap else max(0.0, 1 - (fee - cap) / cap)

def distance_fit(dist, limit, days):
    if dist is None: return 0.5
    if days <= 2: limit = min(limit, 400) if limit else 400
    if limit is None: return 1.0
    return 1.0 if dist <= limit else max(0.0, 1 - (dist - limit) / limit)

def comfort_fit(text, easy_needed):
    if not easy_needed: return 1.0
    t = str(text).lower()
    return 0.3 if any(w in t for w in STRENUOUS_WORDS) else 1.0

def normalize01(arr):
    arr = np.asarray(arr, dtype=float)
    return arr / arr.max() if arr.size and arr.max() > 0 else np.zeros_like(arr)

def mmr_select(total, sim_matrix, k, lam=0.75):
    chosen, remaining = [], list(range(len(total)))
    while remaining and len(chosen) < k:
        def mmr(i):
            penalty = max((sim_matrix[i][j] for j in chosen), default=0)
            return lam * total[i] - (1 - lam) * penalty
        best = max(remaining, key=mmr)
        chosen.append(best)
        remaining.remove(best)
    return chosen

def exact_visit_order(points):
    n = len(points) - 1
    if n <= 0: return []
    dist = [[haversine_km(points[i], points[j]) for j in range(n + 1)] for i in range(n + 1)]
    inf = float("inf")
    size = 1 << n
    cost = [[inf] * n for _ in range(size)]
    parent = [[-1] * n for _ in range(size)]
    for j in range(n): cost[1 << j][j] = dist[0][j + 1]

    for mask in range(size):
        for j in range(n):
            if not (mask & (1 << j)): continue
            here = cost[mask][j]
            if here == inf: continue
            for nxt in range(n):
                if mask & (1 << nxt): continue
                new_cost = here + dist[j + 1][nxt + 1]
                new_mask = mask | (1 << nxt)
                if new_cost < cost[new_mask][nxt]:
                    cost[new_mask][nxt] = new_cost
                    parent[new_mask][nxt] = j

    full_mask = size - 1
    best_last = min(range(n), key=lambda j: cost[full_mask][j] + dist[j + 1][0])
    order, curr_mask, curr_node = [], full_mask, best_last
    while curr_node != -1:
        order.append(curr_node + 1)
        prev_node = parent[curr_mask][curr_node]
        curr_mask ^= (1 << curr_node)
        curr_node = prev_node
    order.reverse()
    return order

# =====================================================
# DATA PREPARATION
# =====================================================
def clean_text(value):
    if pd.isna(value): return ""
    return str(value).strip()

def find_column(df, names):
    def normalize(val): return re.sub(r"[^a-z0-9]", "", str(val).lower())
    lookup = {normalize(col): col for col in df.columns}
    for name in names:
        match = lookup.get(normalize(name))
        if match is not None: return match
    return None

def prepare_data(raw):
    df = raw.copy()
    name_col = find_column(df, ["Name", "Place", "Destination"])
    state_col = find_column(df, ["State", "Region"])
    city_col = find_column(df, ["City", "District"])
    type_col = find_column(df, ["Type", "Category"])
    desc_col = find_column(df, ["Significance", "Description"])
    fee_col = find_column(df, ["Entrance Fee in INR", "Fee"])
    image_col = find_column(df, ["Image", "Image URL"])
    opening_col = find_column(df, ["Opening Time", "Timings"])
    closing_col = find_column(df, ["Closing Time"])
    weekly_col = find_column(df, ["Weekly Off"])
    best_col = find_column(df, ["Best Time to visit"])
    lat_col = find_column(df, ["Latitude", "Lat"])
    lon_col = find_column(df, ["Longitude", "Lon"])

    df["_name"] = df[name_col].apply(clean_text) if name_col else ""
    df["_state"] = df[state_col].apply(clean_text) if state_col else "Not specified"
    df["_city"] = df[city_col].apply(clean_text) if city_col else ""
    df["_type"] = df[type_col].apply(clean_text) if type_col else "Attraction"
    df["_description"] = df[desc_col].apply(clean_text) if desc_col else ""

    if fee_col:
        fee = df[fee_col].astype(str).str.replace(",", "").str.replace(r"[^0-9.]", "", regex=True)
        df["_fee"] = pd.to_numeric(fee, errors="coerce")
    else:
        df["_fee"] = np.nan

    # Google rating and review volume used by the AI popularity score.
    rating_col = find_column(df, ["Google review rating", "Google Rating", "Rating"])
    reviews_col = find_column(
        df,
        ["Number of google review in lakhs", "Google Reviews", "Review Count", "Number of Reviews"]
    )

    df["_rating"] = (
        pd.to_numeric(df[rating_col], errors="coerce")
        if rating_col else np.nan
    )
    df["_reviews"] = (
        pd.to_numeric(df[reviews_col], errors="coerce")
        if reviews_col else np.nan
    )

    df["_image"] = df[image_col].apply(clean_text) if image_col else ""
    df["_latitude"] = pd.to_numeric(df[lat_col], errors="coerce") if lat_col else np.nan
    df["_longitude"] = pd.to_numeric(df[lon_col], errors="coerce") if lon_col else np.nan

    df["_opening_time"] = df[opening_col].apply(clean_text) if opening_col else "9:00 AM"
    df["_closing_time"] = df[closing_col].apply(clean_text) if closing_col else "5:00 PM"
    df["_weekly_off"] = df[weekly_col].apply(clean_text) if weekly_col else "Open daily"
    df["_best_time"] = df[best_col].apply(clean_text) if best_col else "October to March"

    # Extra columns from the v2 dataset: visit duration and nearest transport.
    hours_col = find_column(df, ["time needed to visit in hrs", "Time Needed", "Visit Duration"])
    df["_hours_needed"] = pd.to_numeric(df[hours_col], errors="coerce") if hours_col else np.nan
    dslr_col = find_column(df, ["DSLR Allowed"])
    df["_dslr"] = df[dslr_col].apply(clean_text) if dslr_col else ""
    transport_col = find_column(df, ["Nearest Metro / Transport"])
    tver_col = find_column(df, ["Transport Verified"])
    df["_nearest_transport"] = df[transport_col].apply(clean_text) if transport_col else ""
    df["_transport_verified"] = (
        df[tver_col].astype(str).str.strip().str.lower().str.startswith("yes") if tver_col else False
    )

    df = df[df["_name"] != ""].drop_duplicates(subset=["_name", "_city", "_state"]).reset_index(drop=True)
    df["_search_text"] = (df["_name"] + " " + df["_state"] + " " + df["_city"] + " " + df["_type"] + " " + df["_description"]).str.lower()
    return df

@st.cache_data(show_spinner=False)
def load_data(uploaded=None, extra_version=0):
    raw = None
    if uploaded is not None:
        raw = pd.read_csv(uploaded)
    else:
        for path in DATA_PATHS:
            if path.exists():
                raw = pd.read_csv(path)
                break
    if raw is None:
        raise FileNotFoundError("No tourist dataset found.")
    extra = load_extra_places()
    if extra is not None and not extra.empty:
        raw = pd.concat([raw, align_extra_to_main(raw, extra)], ignore_index=True)
    return prepare_data(raw)

# =====================================================
# TRIP INTELLIGENCE HELPERS
# (explorer map, day clusters, weather, chatbot, evaluation, PDF)
# =====================================================
DAY_COLORS = ["#DC2626", "#2563EB", "#16A34A", "#7C3AED", "#EA580C", "#0E7490", "#BE185D"]
DAY_PIN_COLORS = ["red", "blue", "green", "purple", "orange", "cadetblue", "pink"]
WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]

PIN_COLORS = [
    ("fort", "red"), ("palace", "darkred"), ("temple", "purple"), ("beach", "lightblue"),
    ("park", "green"), ("garden", "green"), ("wildlife", "darkgreen"), ("museum", "orange"),
    ("monument", "darkred"), ("heritage", "red"), ("lake", "blue"), ("hill", "darkgreen"),
    ("waterfall", "cadetblue"), ("church", "purple"), ("mosque", "purple"),
]
VISIT_HOURS = [
    ("trek", 3.5), ("safari", 3.0), ("wildlife", 3.0), ("fort", 2.0), ("palace", 2.0),
    ("museum", 1.5), ("beach", 2.0), ("park", 1.5), ("garden", 1.5), ("lake", 1.5),
    ("waterfall", 1.5), ("temple", 1.0), ("monument", 1.0),
]
WMO_TEXT = {
    0: "Clear sky", 1: "Mostly clear", 2: "Partly cloudy", 3: "Overcast", 45: "Fog", 48: "Fog",
    51: "Light drizzle", 53: "Drizzle", 55: "Heavy drizzle", 61: "Light rain", 63: "Rain",
    65: "Heavy rain", 80: "Rain showers", 81: "Rain showers", 82: "Violent showers",
    95: "Thunderstorm", 96: "Thunderstorm with hail", 99: "Thunderstorm with hail",
}
RAIN_CODES = {51, 53, 55, 61, 63, 65, 80, 81, 82, 95, 96, 99}


def pin_color(text):
    t = str(text).lower()
    for key, color in PIN_COLORS:
        if key in t:
            return color
    return "gray"


def haversine_np(lat, lon, lats, lons):
    lat1, lon1 = math.radians(lat), math.radians(lon)
    lat2 = np.radians(np.asarray(lats, dtype=float))
    lon2 = np.radians(np.asarray(lons, dtype=float))
    h = np.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def nearby_places(pool, row, radius_km, types=None, limit=50):
    """Places within radius_km of `row`, nearest first, with a _dist_km column."""
    lat, lon = row.get("_latitude"), row.get("_longitude")
    empty = pool.iloc[0:0].copy()
    empty["_dist_km"] = pd.Series(dtype=float)
    if pd.isna(lat) or pd.isna(lon):
        return empty
    cand = pool.dropna(subset=["_latitude", "_longitude"]).copy()
    cand = cand[~((cand["_name"] == row.get("_name")) & (cand["_city"] == row.get("_city")))]
    if types:
        cand = cand[cand["_type"].isin(types)]
    if cand.empty:
        return empty
    cand["_dist_km"] = haversine_np(lat, lon, cand["_latitude"], cand["_longitude"])
    return cand[cand["_dist_km"] <= radius_km].sort_values("_dist_km").head(limit)


GEO_FILL_PATH = BASE_DIR / "geocode_fill.json"


def load_geo_fill():
    try:
        return json.loads(GEO_FILL_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_geo_fill(name, city, state, lat, lon):
    """Remember an exact location found by lookup so it survives restarts."""
    fill = load_geo_fill()
    fill[f"{name}|{city}|{state}"] = [round(float(lat), 6), round(float(lon), 6)]
    try:
        GEO_FILL_PATH.write_text(json.dumps(fill, ensure_ascii=False), encoding="utf-8")
    except Exception:
        pass


def apply_geo_fill(frame):
    fill = load_geo_fill()
    if not fill:
        return frame
    frame = frame.copy()
    keys = frame["_name"].astype(str) + "|" + frame["_city"].astype(str) + "|" + frame["_state"].astype(str)
    for i, k in keys.items():
        pos = fill.get(k)
        if pos and (pd.isna(frame.at[i, "_latitude"]) or pd.isna(frame.at[i, "_longitude"])):
            frame.at[i, "_latitude"], frame.at[i, "_longitude"] = pos
    return frame


def _city_centres():
    """Median coordinates of the known places in each (state, city); used for approximate pins."""
    known = df.dropna(subset=["_latitude", "_longitude"])
    return {k: (float(g["_latitude"].median()), float(g["_longitude"].median())) for k, g in known.groupby(["_state", "_city"])}


def fill_exact_locations(frame):
    """Look up exact coordinates for places without them. Returns how many were found."""
    centres, found = _city_centres(), 0
    total = max(len(frame), 1)
    bar = st.progress(0.0, text="Looking up exact locations (about a second per place)...")
    try:
        for n, (_, r) in enumerate(frame.iterrows(), start=1):
            bar.progress(n / total, text=f"Looking up locations... {n} of {total}")
            xy = geocode_cached(r["_name"], r["_city"], r["_state"])
            if not xy:
                continue
            lat, lon = float(xy[0]), float(xy[1])
            if not (6 <= lat <= 38 and 68 <= lon <= 98):
                continue
            c = centres.get((r["_state"], r["_city"]))
            if c and haversine_km(c, (lat, lon)) > 80:
                continue  # geocoder found a same-named place elsewhere
            save_geo_fill(r["_name"], r["_city"], r["_state"], lat, lon)
            found += 1
    finally:
        bar.empty()
    return found


def render_explorer_map(places, center_row=None, radius_km=None, height=520, key="explorer_map"):
    """Clustered pin map. Returns (mapped, missing_coordinates)."""
    exact = places.dropna(subset=["_latitude", "_longitude"]).copy()
    exact["_approx"] = False
    no_xy = places[places["_latitude"].isna() | places["_longitude"].isna()]
    centres = _city_centres() if len(no_xy) else {}
    approx_rows = []
    for _, r in no_xy.iterrows():
        c = centres.get((r.get("_state"), r.get("_city")))
        if not c:
            continue
        # Small fixed offset so several places in one city do not sit exactly on top of each other.
        h = int(hashlib.md5(str(r["_name"]).encode("utf-8")).hexdigest()[:8], 16)
        a = r.copy()
        a["_latitude"] = c[0] + (((h % 200) - 100) / 100) * 0.012
        a["_longitude"] = c[1] + ((((h // 200) % 200) - 100) / 100) * 0.012
        a["_approx"] = True
        approx_rows.append(a)
    pts = pd.concat([exact, pd.DataFrame(approx_rows)], ignore_index=True) if approx_rows else exact
    missing = len(no_xy) - len(approx_rows)
    st.session_state["_map_approx"] = len(approx_rows)
    if pts.empty and center_row is None:
        st.warning("None of these places have GPS coordinates yet, so they cannot be mapped.")
        return 0, missing

    if center_row is not None and pd.notna(center_row.get("_latitude")):
        centre = [float(center_row["_latitude"]), float(center_row["_longitude"])]
    else:
        centre = [float(pts["_latitude"].mean()), float(pts["_longitude"].mean())]
    m = folium.Map(location=centre, zoom_start=6, tiles="OpenStreetMap")
    cluster = MarkerCluster().add_to(m)

    for _, r in pts.iterrows():
        fee = r.get("_fee")
        rating = r.get("_rating")
        extra = f"<br><b>Distance:</b> {r['_dist_km']:.1f} km" if "_dist_km" in r and pd.notna(r.get("_dist_km")) else ""
        is_approx = r.get("_approx") is True
        note = "<br><i>Approximate: placed at the city centre</i>" if is_approx else ""
        popup = (
            f"<div style='font-family:Arial;width:190px'><b>{html.escape(str(r['_name']))}</b><br>"
            f"{html.escape(str(r.get('_city', '')))}, {html.escape(str(r.get('_state', '')))}<br>"
            f"<b>Type:</b> {html.escape(str(r.get('_type', '')))}<br>"
            f"<b>Fee:</b> {'₹%.0f' % fee if pd.notna(fee) else 'not listed'}<br>"
            f"<b>Rating:</b> {'%.1f' % rating if pd.notna(rating) else 'n/a'}{extra}{note}</div>"
        )
        folium.Marker(
            [float(r["_latitude"]), float(r["_longitude"])],
            popup=folium.Popup(popup, max_width=240),
            tooltip=str(r["_name"]),
            icon=folium.Icon(color="gray" if is_approx else pin_color(r.get("_type", "")), icon="info-sign"),
        ).add_to(cluster)

    bounds = [[float(pts["_latitude"].min()), float(pts["_longitude"].min())],
              [float(pts["_latitude"].max()), float(pts["_longitude"].max())]] if not pts.empty else None
    if center_row is not None and pd.notna(center_row.get("_latitude")):
        folium.Marker(
            centre, tooltip=f"You are here: {center_row['_name']}",
            icon=folium.Icon(color="black", icon="star"),
        ).add_to(m)
        if radius_km:
            folium.Circle(centre, radius=radius_km * 1000, color="#D49B28", fill=True, fill_opacity=0.08).add_to(m)
        if bounds is None:
            bounds = [centre, centre]
        else:
            bounds[0][0] = min(bounds[0][0], centre[0]); bounds[0][1] = min(bounds[0][1], centre[1])
            bounds[1][0] = max(bounds[1][0], centre[0]); bounds[1][1] = max(bounds[1][1], centre[1])
    if bounds:
        m.fit_bounds(bounds, max_zoom=12)
    st_folium(m, width="100%", height=height, returned_objects=[], key=key)
    return len(pts), missing


# ---------- itinerary: day clusters, schedule, weather ----------
def visit_hours(type_text, name_text=""):
    t = f"{type_text} {name_text}".lower()
    for key, hrs in VISIT_HOURS:
        if key in t:
            return hrs
    return 1.5


def parse_clock(value):
    t = str(value).strip().lower()
    m = re.search(r"(\d{1,2})(?::(\d{2}))?\s*([ap])\.?m\.?", t) or re.search(r"(\d{1,2}):(\d{2})()", t)
    if not m:
        return None
    h, mi, ap = int(m.group(1)), int(m.group(2) or 0), m.group(3)
    if h > 24 or mi > 59:
        return None
    if ap == "p" and h < 12:
        h += 12
    if ap == "a" and h == 12:
        h = 0
    return h + mi / 60


def fmt_clock(h):
    h = h % 24
    hh, mm = int(h), int(round((h - int(h)) * 60))
    if mm == 60:
        hh, mm = (hh + 1) % 24, 0
    suffix = "AM" if hh < 12 else "PM"
    return f"{(hh % 12) or 12}:{mm:02d} {suffix}"


def place_hours(row):
    o = str(row.get("_opening_time", "")).lower()
    c = str(row.get("_closing_time", "")).lower()
    if ("24" in o and "hour" in o) or ("24" in c and "hour" in c) or "all day" in o:
        return 0.0, 24.0
    return parse_clock(o), parse_clock(c)


def is_closed_on(weekly_off, date):
    t = str(weekly_off).lower()
    if not t or re.search(r"no weekly|open daily|\bnone\b|\bnil\b|all days|7 days|\bn/?a\b", t):
        return False
    return WEEKDAYS[date.weekday()] in t


def cluster_into_days(stops, k):
    """Group stops into k geographic clusters of roughly equal size."""
    n = len(stops)
    k = max(1, min(int(k), n))
    if k == 1 or n <= 1:
        return np.zeros(n, dtype=int)
    coords = stops[["_latitude", "_longitude"]].to_numpy(dtype=float)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(coords)
    labels, centers = km.labels_.copy(), km.cluster_centers_
    cap = math.ceil(n / k)
    for _ in range(n * k + 1):
        counts = np.bincount(labels, minlength=k)
        big = int(np.argmax(counts))
        if counts[big] <= cap:
            break
        best = None
        for i in np.where(labels == big)[0]:
            for c in range(k):
                if c == big or counts[c] >= cap:
                    continue
                d = haversine_km(coords[i], centers[c])
                if d > 50:  # never balance by sending a stop to a far-away day
                    continue
                if best is None or d < best[0]:
                    best = (d, i, c)
        if best is None:
            break
        labels[best[1]] = best[2]
    return labels


SLOT_ORDER = {"morning": 0, "afternoon": 1, "evening": 2}
SLOT_WINDOWS = {"morning": (5.0, 12.0), "afternoon": (12.0, 17.0), "evening": (16.0, 22.0)}


def best_slot(text):
    """Best time of day from the dataset: morning, afternoon or evening (None if unknown)."""
    t = str(text).lower()
    found = [(t.find(k), k) for k in ("morning", "afternoon", "evening", "sunset", "night") if k in t]
    if not found:
        return None
    key = min(found)[1]
    return "evening" if key in ("sunset", "night") else key


def build_day_plan(stops, days, start_hour=9.0, day_limit_h=8.0, speed_kmh=25.0, start_date=None):
    """Cluster stops into days, order each day by shortest route, then add a timed schedule."""
    stops = stops.dropna(subset=["_latitude", "_longitude"]).reset_index(drop=True)
    n = len(stops)
    if n == 0:
        return []
    labels = cluster_into_days(stops, days)
    coords = [(float(a), float(b)) for a, b in zip(stops["_latitude"], stops["_longitude"])]
    home = coords[0]
    groups = []
    for c in sorted(set(int(x) for x in labels)):
        idx = [i for i in range(n) if labels[i] == c]
        cen = (float(np.mean([coords[i][0] for i in idx])), float(np.mean([coords[i][1] for i in idx])))
        groups.append((idx, cen))
    groups.sort(key=lambda g: haversine_km(home, g[1]))

    plan, prev_xy = [], home
    for d, (idx, cen) in enumerate(groups):
        first = min(idx, key=lambda i: haversine_km(prev_xy, coords[i]))
        rest = [i for i in idx if i != first]
        pts = [coords[first]] + [coords[i] for i in rest]
        order = exact_visit_order(pts) if len(pts) > 1 else []
        seq = [first] + [rest[j - 1] for j in order]
        # Prefer each place's best time of day when it barely lengthens the route.
        slots = {i: best_slot(stops.iloc[i].get("_best_time", "")) for i in seq}
        if len(seq) > 1 and any(slots.values()):
            slot_seq = sorted(seq, key=lambda i: SLOT_ORDER.get(slots[i], 1))
            km_a = sum(haversine_km(coords[a], coords[b]) for a, b in zip(seq, seq[1:]))
            km_b = sum(haversine_km(coords[a], coords[b]) for a, b in zip(slot_seq, slot_seq[1:]))
            if km_b <= km_a * 1.25 + 2:
                seq = slot_seq
        date = start_date + datetime.timedelta(days=d) if start_date else None

        t, prev, km_total, lunch_done, items = start_hour, None, 0.0, False, []
        for i in seq:
            row = stops.iloc[i]
            notes, travel_km = [], 0.0
            if prev is not None:
                travel_km = haversine_km(prev, coords[i]) * 1.3
                t += travel_km / speed_kmh
                km_total += travel_km
                if travel_km / speed_kmh > 2:
                    notes.append(f"Long transfer of about {travel_km / speed_kmh:.1f} h from the previous stop; consider a separate day")
            open_h, close_h = place_hours(row)
            if open_h is not None and t < open_h:
                notes.append(f"Opens at {fmt_clock(open_h)}, so arrival is moved to opening time")
                t = open_h
            lunch_before = False
            if not lunch_done and t >= 13:
                t, lunch_done, lunch_before = t + 1.0, True, True
            _hn = row.get("_hours_needed")
            hrs = float(_hn) if pd.notna(_hn) and float(_hn) > 0 else visit_hours(row["_type"], row["_name"])
            arrive, leave = t, t + hrs
            if close_h is not None and close_h > (open_h or 0) and leave > close_h:
                notes.append(f"Closes at {fmt_clock(close_h)}: shorten the visit or go earlier")
            if date and is_closed_on(row.get("_weekly_off", ""), date):
                notes.append(f"Weekly off on {date.strftime('%A')}: consider another day")
            text = f"{row['_type']} {row['_name']}".lower()
            fee = row.get("_fee")
            items.append({
                "name": row["_name"], "city": row.get("_city", ""), "type": row["_type"],
                "fee": float(fee) if pd.notna(fee) else None,
                "lat": coords[i][0], "lon": coords[i][1],
                "arrive": arrive, "leave": leave, "hours": hrs,
                "travel_km": travel_km, "notes": notes, "lunch_before": lunch_before,
                "outdoor": any(w in text for w in OUTDOOR_WORDS),
                "slot": slots.get(i), 
                "slot_ok": slots.get(i) is None or SLOT_WINDOWS[slots[i]][0] <= (arrive % 24) <= SLOT_WINDOWS[slots[i]][1],
                "transport": str(row.get("_nearest_transport", "") or ""),
                "transport_verified": bool(row.get("_transport_verified")) if isinstance(row.get("_transport_verified"), (bool, np.bool_)) else False,
                "approx_xy": bool(row.get("_xy_approx")) if isinstance(row.get("_xy_approx"), (bool, np.bool_)) else False,
            })
            t, prev = leave, coords[i]
        plan.append({
            "day": d + 1, "date": date, "stops": items, "centroid": cen,
            "km": km_total, "end": t, "overloaded": (t - start_hour) > day_limit_h,
        })
        prev_xy = coords[seq[-1]]
    return plan


def render_day_plan_map(plan, key="day_plan_map"):
    pts = [(s["lat"], s["lon"]) for day in plan for s in day["stops"]]
    if not pts:
        return
    m = folium.Map(location=[np.mean([p[0] for p in pts]), np.mean([p[1] for p in pts])], zoom_start=11)
    for day in plan:
        color_idx = (day["day"] - 1) % len(DAY_COLORS)
        path = []
        for n, s in enumerate(day["stops"], start=1):
            path.append((s["lat"], s["lon"]))
            folium.Marker(
                [s["lat"], s["lon"]],
                tooltip=f"Day {day['day']} · Stop {n}: {s['name']}",
                popup=f"Day {day['day']} · Stop {n}<br>{html.escape(str(s['name']))}<br>{fmt_clock(s['arrive'])}",
                icon=folium.Icon(color=DAY_PIN_COLORS[color_idx], icon="info-sign"),
            ).add_to(m)
        if len(path) > 1:
            folium.PolyLine(path, color=DAY_COLORS[color_idx], weight=4, opacity=0.85).add_to(m)
    m.fit_bounds([[min(p[0] for p in pts), min(p[1] for p in pts)],
                  [max(p[0] for p in pts), max(p[1] for p in pts)]], max_zoom=13)
    st_folium(m, width="100%", height=460, returned_objects=[], key=key)


@st.cache_data(ttl=1800, show_spinner=False)
def get_daily_forecast(lat, lon, start_iso, end_iso):
    try:
        response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon, "timezone": "auto",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                "start_date": start_iso, "end_date": end_iso,
            },
            timeout=8,
        )
        if response.status_code == 200:
            return response.json().get("daily")
    except Exception:
        pass
    return None


def weather_for_day(lat, lon, date):
    """Forecast for one day, or None when unavailable (Open-Meteo covers about 16 days)."""
    today = datetime.date.today()
    if date is None or date < today or date > today + datetime.timedelta(days=15):
        return None
    daily = get_daily_forecast(round(float(lat), 2), round(float(lon), 2), date.isoformat(), date.isoformat())
    if not daily or not daily.get("time"):
        return None
    try:
        codes = daily.get("weather_code") or daily.get("weathercode") or [None]
        return {
            "tmax": daily["temperature_2m_max"][0], "tmin": daily["temperature_2m_min"][0],
            "rain": daily["precipitation_probability_max"][0], "code": codes[0],
        }
    except (KeyError, IndexError, TypeError):
        return None


def weather_advice(w, stops):
    rain = w.get("rain") or 0
    tmax = w.get("tmax")
    outdoor = [s["name"] for s in stops if s["outdoor"]]
    tips = []
    if rain >= 60 or w.get("code") in RAIN_CODES:
        tip = f"Rain is likely ({rain:.0f}%). Do covered stops first and keep an umbrella ready."
        if outdoor:
            tip += f" Outdoor stops at risk: {', '.join(outdoor)}."
        tips.append(tip)
    if tmax is not None and tmax >= 36:
        tips.append("Very hot day. Visit outdoor stops before 11 AM or after 4 PM and carry water.")
    if tmax is not None and tmax <= 8:
        tips.append("Cold day. Carry warm layers.")
    return tips


def weather_line(w):
    desc = WMO_TEXT.get(w.get("code"), "Mixed conditions")
    tmin, tmax = w.get("tmin"), w.get("tmax")
    temp = f"{tmin:.0f}–{tmax:.0f}°C" if tmin is not None and tmax is not None else "temperature n/a"
    rain = f"{w['rain']:.0f}% rain chance" if w.get("rain") is not None else "rain chance n/a"
    return f"{desc} · {temp} · {rain}"


# ---------- chatbot grounded in the dataset ----------
def quick_trip_cost(tier, days, people, entry_pp):
    r = BUDGET_PRESETS[tier]
    rooms = math.ceil(people / 2)
    return r["hotel"] * days * rooms + (r["food"] + r["local"]) * days * people + entry_pp * people


def parse_money(text):
    t = text.lower().replace(",", "")
    mult = {"k": 1e3, "thousand": 1e3}
    patterns = [
        r"(?:₹|rs\.?|inr)\s*(\d+(?:\.\d+)?)\s*(k|thousand|lakhs?|lacs?)?",
        r"(?:under|below|within|upto|up to|around|about|max|budget of|budget is|budget)\s*(?:of|is)?\s*(?:₹|rs\.?|inr)?\s*(\d+(?:\.\d+)?)\s*(k|thousand|lakhs?|lacs?)?",
        r"(\d+(?:\.\d+)?)\s*(k|thousand|lakhs?|lacs?)\b",
    ]
    for p in patterns:
        for m in re.finditer(p, t):
            if re.match(r"\s*(day|night|people|person|pax|member|km|adult|friend)", t[m.end():m.end() + 12]):
                continue
            val, suf = float(m.group(1)), m.group(2)
            if suf:
                val *= mult.get(suf, 1e5)
            if val >= 500:
                return val
    return None


def parse_days(text):
    t = text.lower()
    m = re.search(r"(\d+)\s*-?\s*(?:day|days|night|nights)\b", t)
    if m:
        return max(1, min(int(m.group(1)), 14))
    if "weekend" in t:
        return 2
    if re.search(r"\bweek\b", t):
        return 7
    return None


def parse_people(text):
    t = text.lower()
    m = re.search(r"(\d+)\s*(?:people|persons|person|pax|members|friends|adults|travell?ers)", t) or re.search(r"for\s+(\d+)\b(?!\s*(?:day|night|k\b))", t)
    if m:
        return max(1, min(int(m.group(1)), 15))
    if "couple" in t or "honeymoon" in t:
        return 2
    if "solo" in t or "alone" in t:
        return 1
    if "family" in t:
        return 4
    return None


def parse_month(text):
    t = text.lower()
    for i, name in enumerate(MONTHS):
        if re.search(rf"\b{name.lower()}\b", t):
            return i
    for i, ab in enumerate(MONTH_ABBR):
        if ab not in ("may", "mar") and re.search(rf"\b{ab}\b", t):
            return i
    return None


def chat_categories(text):
    t = text.lower()
    return [c for c, kws in CATEGORIES.items() if any(re.search(rf"\b{re.escape(k)}(?:es|s)?\b", t) for k in kws)]


def chat_find_location(text, df):
    t = text.lower()
    state = city = None
    for s in sorted((x for x in df["_state"].unique() if x and x != "Not specified"), key=len, reverse=True):
        if re.search(rf"\b{re.escape(s.lower())}\b", t):
            state = s
            break
    for c in sorted((x for x in df["_city"].unique() if x and len(x) >= 3), key=len, reverse=True):
        if re.search(rf"\b{re.escape(c.lower())}\b", t):
            city = c
            break
    if city and not state:
        state = df.loc[df["_city"] == city, "_state"].iloc[0]
    return state, city


def chat_find_places(text, df, fuzzy=False, limit=3):
    t = text.lower()
    found = []
    for idx, nm in sorted(df["_name"].items(), key=lambda kv: -len(kv[1])):
        low = nm.lower()
        if len(low) >= 4 and re.search(rf"\b{re.escape(low)}\b", t):
            found.append(idx)
            t = t.replace(low, " ")
            if len(found) >= limit:
                break
    if not found and fuzzy:
        words = re.findall(r"[a-z0-9']+", t)
        lows = {nm.lower(): i for i, nm in df["_name"].items()}
        skip = {k for kws in CATEGORIES.values() for k in kws}
        for size in (4, 3, 2, 1):
            for s in range(0, len(words) - size + 1):
                phrase = " ".join(words[s:s + size])
                if len(phrase) < 5 or phrase.rstrip("s") in skip:
                    continue
                m = difflib.get_close_matches(phrase, list(lows), n=1, cutoff=0.82)
                if m and lows[m[0]] not in found:
                    found.append(lows[m[0]])
            if found:
                break
    return df.loc[found[:limit]]


def chat_intent(t):
    if re.search(r"\b(compare|versus|vs\.?)\b", t):
        return "compare"
    if re.search(r"\b(plan|itinerary|schedule)\b", t) or (
        re.search(r"\b\d+\s*-?\s*(?:day|days|night|nights)\b", t) and re.search(r"\b(trip|tour|visit|travel|holiday)\b", t)
    ):
        return "plan"
    if re.search(r"\b(near|nearby|around|close to)\b|within \d+\s*km", t):
        return "nearby"
    if re.search(r"best time|when (?:to|should)|which month|what season|peak season", t):
        return "besttime"
    if re.search(r"\b(fee|fees|ticket|tickets|entry|entrance|price)\b", t):
        return "fee"
    if re.search(r"\b(open|opening|close|closing|timing|timings|hours|weekly off)\b", t):
        return "timing"
    if re.search(r"\b(free|cheap|cheapest|low cost|inexpensive|no fee)\b", t):
        return "cheap"
    if re.search(r"how much|\bcost\b|expense|estimate|\bbudget\b", t):
        return "cost"
    if re.search(r"\b(about|tell me|describe|where is|what is)\b", t):
        return "about"
    if re.search(r"\b(suggest|recommend|best|top|popular|famous|must|places|visit|see|explore)\b", t):
        return "recommend"
    return "unknown"


def chat_rank(pool, month_idx=None):
    if pool.empty:
        return pool.assign(_chat_score=pd.Series(dtype=float))
    rating = pd.to_numeric(pool["_rating"], errors="coerce")
    med = rating.median()
    med = 3.0 if pd.isna(med) else med
    r = rating.fillna(med).clip(0, 5).to_numpy(dtype=float) / 5
    rev = np.log1p(pd.to_numeric(pool["_reviews"], errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=float))
    score = 0.6 * r + 0.4 * normalize01(rev)
    if month_idx is not None:
        season = season_scores(pool, month_idx)
        score = 0.8 * score + 0.2 * season
    out = pool.copy()
    out["_chat_score"] = score
    return out.sort_values("_chat_score", ascending=False)


def _fee_text(fee):
    return "entry fee not listed" if pd.isna(fee) else ("free entry" if fee == 0 else f"entry ₹{fee:,.0f}")


def place_line(row):
    rating = row.get("_rating")
    rate = f" · ⭐ {rating:.1f}" if pd.notna(rating) else ""
    where = ", ".join(x for x in [row.get("_city", ""), row.get("_state", "")] if x)
    return f"**{row['_name']}** ({where}) · {row.get('_type', '')}{rate} · {_fee_text(row.get('_fee'))}"


def chat_answer(prompt, df_all, df_active):
    """Rule-based assistant. Every fact comes from the dataset; returns (markdown, places_used)."""
    t = prompt.lower().strip()
    intent = chat_intent(t)
    state, city = chat_find_location(prompt, df_all)
    place_specific = intent in ("fee", "timing", "about", "compare", "nearby", "besttime", "unknown")
    places = chat_find_places(prompt, df_all, fuzzy=place_specific)
    cats = chat_categories(prompt)
    month_idx = parse_month(prompt)
    days, people, budget = parse_days(prompt), parse_people(prompt), parse_money(prompt)

    base = df_all if (state or city) else df_active
    pool = base
    if city:
        pool = pool[pool["_city"] == city]
    elif state:
        pool = pool[pool["_state"] == state]
    where_txt = city or state or "your selected area"

    direct = sorted({k for c in cats for k in CATEGORIES[c] if re.search(rf"\b{re.escape(k)}(?:es|s)?\b", t)})

    def cat_filter(p):
        if not cats:
            return p, ""
        pat = "|".join(re.escape(k) for k in (direct or [k for c in cats for k in CATEGORIES[c]]))
        m = p["_search_text"].str.contains(pat, case=False, na=False, regex=True)
        if m.any():
            return p[m], ""
        return p, f"_No exact {', '.join(cats)} match in the dataset for {where_txt}; showing the closest overall._\n\n"

    if intent == "compare" and len(places) >= 2:
        rows = places.head(3)
        head = "| | " + " | ".join(rows["_name"]) + " |\n|---|" + "---|" * len(rows) + "\n"
        def cell(fn): return "| " + " | ".join(fn(r) for _, r in rows.iterrows()) + " |\n"
        body = (
            "| Location " + cell(lambda r: f"{r['_city']}, {r['_state']}") +
            "| Type " + cell(lambda r: str(r["_type"])) +
            "| Rating " + cell(lambda r: f"{r['_rating']:.1f}" if pd.notna(r["_rating"]) else "n/a") +
            "| Entry fee " + cell(lambda r: f"₹{r['_fee']:,.0f}" if pd.notna(r["_fee"]) else "not listed") +
            "| Timings " + cell(lambda r: f"{r['_opening_time']} to {r['_closing_time']}") +
            "| Best time " + cell(lambda r: str(r["_best_time"]))
        )
        return head + body, rows

    if intent in ("fee", "timing", "about") and len(places):
        lines = []
        for _, r in places.iterrows():
            if intent == "fee":
                lines.append(f"- **{r['_name']}**: {_fee_text(r['_fee'])}.")
            elif intent == "timing":
                off = r["_weekly_off"] or "not listed"
                lines.append(f"- **{r['_name']}**: {r['_opening_time']} to {r['_closing_time']}; weekly off: {off}.")
            else:
                desc = str(r["_description"])[:280]
                lines.append(f"- {place_line(r)}\n  Best time: {r['_best_time']}. {desc}")
        return "\n".join(lines), places

    if intent == "nearby":
        radius = 50
        m = re.search(r"(\d+)\s*km", t)
        if m:
            radius = max(5, min(int(m.group(1)), 500))
        ref = None
        if len(places):
            ref = places.iloc[0]
        elif city:
            g = df_all[df_all["_city"] == city].dropna(subset=["_latitude", "_longitude"])
            if not g.empty:
                ref = pd.Series({"_name": f"{city} centre", "_city": "", "_latitude": g["_latitude"].mean(), "_longitude": g["_longitude"].mean()})
        if ref is None:
            return "Tell me which place or city to search around, for example _places near Jaipur within 100 km_.", df_all.iloc[0:0]
        near, note = cat_filter(df_all.dropna(subset=["_latitude", "_longitude"]))
        near = nearby_places(near, ref, radius).head(8)
        if near.empty:
            return f"I found nothing within {radius} km of {ref['_name']} in the dataset.", near
        lines = [f"- {place_line(r)} · {r['_dist_km']:.0f} km away" for _, r in near.iterrows()]
        return note + f"Places within {radius} km of **{ref['_name']}**:\n" + "\n".join(lines), near

    if intent == "besttime":
        if len(places):
            lines = [f"- **{r['_name']}**: {r['_best_time']}" for _, r in places.iterrows()]
            return "\n".join(lines), places
        top = chat_rank(pool).head(40)
        if top.empty:
            return "I couldn't find places for that area in the dataset.", top
        counts = [(MONTHS[i], int((season_scores(top, i) == 1.0).sum())) for i in range(12)]
        counts.sort(key=lambda x: -x[1])
        best = [f"{m} ({c} of {len(top)} top places)" for m, c in counts[:3] if c > 0]
        if not best:
            return f"The dataset has no clear best-time information for {where_txt}.", top
        return f"Based on the best-time notes for the top places in **{where_txt}**, the strongest months are: " + ", ".join(best) + ".", top.head(5)

    if intent == "cheap":
        p, note = cat_filter(pool)
        listed = p[p["_fee"].notna()]
        free = listed[listed["_fee"] == 0]
        out = (free if len(free) else listed.sort_values("_fee")).head(6)
        if out.empty:
            return f"No fee information is listed for places in {where_txt}.", out
        unlisted = int(p["_fee"].isna().sum())
        head = "Free places" if len(free) else "Lowest listed entry fees"
        tail = f"\n\n_{unlisted} place(s) here have no fee listed, so they are not included._" if unlisted else ""
        return note + f"{head} in **{where_txt}**:\n" + "\n".join(f"- {place_line(r)}" for _, r in out.iterrows()) + tail, out

    if intent == "plan":
        n_days, n_people = days or 3, people or 2
        p, note = cat_filter(pool)
        ranked = chat_rank(p, month_idx)
        if ranked.empty:
            return "I couldn't find places for that request in the dataset.", ranked
        need = min(len(ranked), n_days * 3)
        pick_note = ""
        if not (state or city) and ranked["_state"].nunique() > 1:
            best_state = max(ranked["_state"].unique(), key=lambda s: ranked[ranked["_state"] == s].head(need)["_chat_score"].sum())
            ranked = ranked[ranked["_state"] == best_state]
            where_txt = best_state
            pick_note = f"_You didn't name a place, so I picked **{best_state}**, which has the most highly rated places in your dataset._\n\n"
        chosen = ranked.head(need)
        with_xy = chosen.dropna(subset=["_latitude", "_longitude"])
        lines = []
        if len(with_xy) >= 2:
            plan = build_day_plan(with_xy, n_days)
            for day in plan:
                stops = "; ".join(f"{s['name']} ({fmt_clock(s['arrive'])})" for s in day["stops"])
                lines.append(f"- **Day {day['day']}**: {stops}")
            if len(plan) < n_days:
                lines.append(f"- _Only {len(plan)} day(s) could be filled with the places available._")
        else:
            for i, (_, r) in enumerate(chosen.iterrows(), 1):
                lines.append(f"- **Stop {i}**: {r['_name']} ({r['_city']})")
        entry_pp = float(chosen["_fee"].fillna(0).sum())
        unlisted = int(chosen["_fee"].isna().sum())
        costs = {tier: quick_trip_cost(tier, n_days, n_people, entry_pp) for tier in BUDGET_PRESETS}
        cost_lines = "\n".join(f"- {tier}: ₹{c:,.0f}" for tier, c in costs.items())
        verdict = ""
        if budget:
            fits = [tr for tr in ("Luxury", "Mid-range", "Budget") if costs[tr] <= budget]
            verdict = (f"\n\nWith a budget of ₹{budget:,.0f}, the best tier that fits is **{fits[0]}**." if fits else
                       f"\n\n₹{budget:,.0f} is below even the Budget tier (₹{costs['Budget']:,.0f}). Try fewer days or travellers.")
        month_txt = f" in {MONTHS[month_idx]}" if month_idx is not None else ""
        text = (
            pick_note + note + f"**{n_days}-day plan for {n_people} traveller(s) in {where_txt}{month_txt}**\n" + "\n".join(lines) +
            f"\n\n**Estimated cost** (stay for {math.ceil(n_people / 2)} room(s), food, local travel, entry fees"
            + (f"; {unlisted} stop(s) have no listed fee" if unlisted else "") + f"; intercity travel not included):\n{cost_lines}{verdict}"
            + "\n\nOpen the Itinerary Planner tab to adjust stops, check weather and send fees to the Travel Budget."
        )
        return text, chosen

    if intent == "cost":
        n_days, n_people = days or 3, people or 2
        top = chat_rank(pool).head(n_days * 3)
        entry_pp = float(top["_fee"].fillna(0).sum()) if not top.empty else 0.0
        rows = "\n".join(f"- {tier}: ₹{quick_trip_cost(tier, n_days, n_people, entry_pp):,.0f}" for tier in BUDGET_PRESETS)
        return (f"Estimated cost for **{n_days} day(s), {n_people} traveller(s)** around {where_txt} "
                f"(entry fees from the top {len(top)} places; intercity travel not included):\n{rows}"), top

    if intent in ("recommend", "about") or cats or len(places):
        if len(places) and intent != "recommend":
            return "\n".join(f"- {place_line(r)}" for _, r in places.iterrows()), places
        p, note = cat_filter(pool)
        top = chat_rank(p, month_idx).head(5)
        if top.empty:
            return "I couldn't find matching places in the dataset.", top
        month_txt = f" for {MONTHS[month_idx]}" if month_idx is not None else ""
        return note + f"Top picks in **{where_txt}**{month_txt}:\n" + "\n".join(f"- {place_line(r)}" for _, r in top.iterrows()), top

    return (
        "I can answer from your dataset about:\n"
        "- **Plans and costs**: _plan a 3 day trip in Kerala under ₹20,000_\n"
        "- **Recommendations**: _best beaches in Goa_, _free places in Delhi_\n"
        "- **Place facts**: _entry fee of Taj Mahal_, _timings of Red Fort_, _best time to visit Jaipur_\n"
        "- **Nearby and compare**: _places near Udaipur within 100 km_, _compare Hawa Mahal vs Amber Fort_"
    ), df_all.iloc[0:0]








# ---------- trip summary: markdown and PDF ----------
def trip_summary_markdown(ctx):
    L = [f"# {ctx['title']}", ""]
    L.append(f"Focus: {ctx['focus']}  \nGenerated: {datetime.date.today():%d %b %Y}\n")
    it = ctx.get("itinerary")
    L.append("## Itinerary")
    if it:
        L.append(f"{it['city']} · {len(it['days'])} day(s) · {it['travelers']} traveller(s)\n")
        for d in it["days"]:
            L.append(f"### Day {d['day']}" + (f" · {d['date']}" if d["date"] else ""))
            if d["weather"]:
                L.append(f"Weather: {d['weather']}")
            for s in d["stops"]:
                fee = f"₹{s['fee']:,.0f}" if s["fee"] is not None else "fee not listed"
                L.append(f"- {s['arrive']}–{s['leave']} **{s['name']}** ({s['type']}), {fee}")
                for n in s["notes"]:
                    L.append(f"  - Note: {n}")
            for tip in d["advice"]:
                L.append(f"- Tip: {tip}")
            L.append("")
    else:
        L.append("_No itinerary built yet._\n")
    tr = ctx.get("transport")
    L.append("## Getting there")
    if tr:
        L.append(f"{tr['start']} → {tr['dest']} · about {tr['straight_km']:,.0f} km one way · {tr['travellers']} traveller(s) · suggested: **{tr['best']}**\n")
        L.append("| Mode | Time | Est. cost | Notes |\n|---|---|---|---|")
        L += [f"| {o['mode']} | {o['time']} | ₹{o['cost']:,.0f} | {o['note']} |" for o in tr["options"]]
        L.append("\n_Estimates only; check operators for real timings and fares._\n")
    else:
        L.append("_No travel route checked yet. Use Plan My Trip, then Transport._\n")
    b = ctx.get("budget")
    L.append("## Budget")
    if b:
        L.append(f"Total ₹{b['total']:,.2f} · {b['group']} · {b['method']}\n")
        L.append("| Category | Amount (₹) |\n|---|---|")
        L += [f"| {c} | {a:,.2f} |" for c, a in b["by_cat"]]
        L.append("\n| Member | Fair share | Paid | Balance |\n|---|---|---|---|")
        L += [f"| {r['Member']} | {r['Fair share (₹)']:,.2f} | {r['Actually paid (₹)']:,.2f} | {r['Balance (₹)']:+,.2f} |" for r in b["result"]]
        if b["transfers"]:
            L.append("\n**Settle up**")
            L += [f"- {d} pays {c} ₹{a:,.2f}" for d, c, a in b["transfers"]]
    else:
        L.append("_No budget calculated yet._")
    if ctx.get("favorites"):
        L.append("\n## Saved favourites")
        L += [f"- {f}" for f in ctx["favorites"]]
    return "\n".join(L)


def build_trip_pdf(ctx):
    """Return PDF bytes, or None if reportlab is not installed."""
    try:
        from reportlab.lib import colors
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
    except ImportError:
        return None
    from xml.sax.saxutils import escape

    regular, bold, rupee = "Helvetica", "Helvetica-Bold", False
    try:
        import matplotlib, os
        base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
        reg_path, bold_path = os.path.join(base, "DejaVuSans.ttf"), os.path.join(base, "DejaVuSans-Bold.ttf")
        if os.path.exists(reg_path) and os.path.exists(bold_path):
            pdfmetrics.registerFont(TTFont("DejaVu", reg_path))
            pdfmetrics.registerFont(TTFont("DejaVu-Bold", bold_path))
            regular, bold = "DejaVu", "DejaVu-Bold"
            rupee = 0x20B9 in pdfmetrics.getFont("DejaVu").face.charToGlyph
    except Exception:
        pass

    def clean(s):
        s = str(s).replace("₹", "₹" if rupee else "Rs. ")
        return s if regular == "DejaVu" else s.encode("latin-1", "replace").decode("latin-1")

    def P(text, style):
        return Paragraph(escape(clean(text)), style)

    navy, gold = colors.HexColor("#0B1329"), colors.HexColor("#D49B28")
    h1 = ParagraphStyle("h1", fontName=bold, fontSize=20, leading=26, textColor=navy, spaceAfter=6)
    h2 = ParagraphStyle("h2", fontName=bold, fontSize=14, textColor=navy, spaceBefore=12, spaceAfter=6)
    h3 = ParagraphStyle("h3", fontName=bold, fontSize=11, textColor=colors.HexColor("#1E3A8A"), spaceBefore=8, spaceAfter=3)
    body = ParagraphStyle("body", fontName=regular, fontSize=9.5, leading=13)
    small = ParagraphStyle("small", fontName=regular, fontSize=8.5, leading=11, textColor=colors.HexColor("#475569"))
    cell = ParagraphStyle("cell", fontName=regular, fontSize=8.5, leading=11)

    def table(rows, widths):
        data = [[P(c, cell) for c in r] for r in rows]
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E2E8F0")),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
        ]))
        return t

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm, topMargin=15 * mm, bottomMargin=15 * mm,
                            title=clean(ctx["title"]))
    W = A4[0] - 32 * mm
    s = [P(ctx["title"], h1), P(f"Focus: {ctx['focus']}   |   Generated {datetime.date.today():%d %b %Y}", small), Spacer(1, 4)]

    s.append(P("Itinerary", h2))
    it = ctx.get("itinerary")
    if it:
        s.append(P(f"{it['city']}  |  {len(it['days'])} day(s)  |  {it['travelers']} traveller(s)", body))
        for d in it["days"]:
            s.append(P(f"Day {d['day']}" + (f"  ({d['date']})" if d["date"] else ""), h3))
            if d["weather"]:
                s.append(P(f"Weather: {d['weather']}", small))
            rows = [["Time", "Stop", "Type", "Entry fee", "Notes"]]
            for st_ in d["stops"]:
                fee = f"₹{st_['fee']:,.0f}" if st_["fee"] is not None else "not listed"
                rows.append([f"{st_['arrive']} - {st_['leave']}", st_["name"], st_["type"], fee, "; ".join(st_["notes"])])
            s.append(table(rows, [28 * mm, 48 * mm, 28 * mm, 22 * mm, W - 126 * mm]))
            for tip in d["advice"]:
                s.append(P(f"Tip: {tip}", small))
    else:
        s.append(P("No itinerary built yet.", body))

    s.append(P("Getting there", h2))
    tr = ctx.get("transport")
    if tr:
        s.append(P(f"{tr['start']} to {tr['dest']}  |  about {tr['straight_km']:,.0f} km one way  |  {tr['travellers']} traveller(s)  |  Suggested: {tr['best']}", body))
        s.append(Spacer(1, 4))
        s.append(table([["Mode", "Time", "Est. cost", "Notes"]] + [
            [o["mode"], o["time"], f"₹{o['cost']:,.0f}", o["note"]] for o in tr["options"]
        ], [W * 0.14, W * 0.18, W * 0.18, W * 0.50]))
        s.append(P("Estimates only; check operators for real timings and fares.", small))
    else:
        s.append(P("No travel route checked yet.", body))

    s.append(P("Budget", h2))
    b = ctx.get("budget")
    if b:
        s.append(P(f"Total ₹{b['total']:,.2f}  |  {b['group']}  |  {b['method']}", body))
        s.append(Spacer(1, 4))
        s.append(table([["Category", "Amount (₹)"]] + [[c, f"{a:,.2f}"] for c, a in b["by_cat"]], [W * 0.6, W * 0.4]))
        s.append(Spacer(1, 6))
        s.append(table([["Member", "Fair share", "Paid", "Balance"]] + [
            [r["Member"], f"{r['Fair share (₹)']:,.2f}", f"{r['Actually paid (₹)']:,.2f}", f"{r['Balance (₹)']:+,.2f}"] for r in b["result"]
        ], [W * 0.34, W * 0.22, W * 0.22, W * 0.22]))
        if b["transfers"]:
            s.append(P("Settle up", h3))
            for d_, c_, a_ in b["transfers"]:
                s.append(P(f"{d_} pays {c_} ₹{a_:,.2f}", body))
    else:
        s.append(P("No budget calculated yet.", body))

    if ctx.get("favorites"):
        s.append(P("Saved favourites", h2))
        for f in ctx["favorites"]:
            s.append(P(f"- {f}", body))
    doc.build(s)
    return buf.getvalue()


# =====================================================
# DATA GAPS: user-added places, Wikipedia images, live transport (OpenStreetMap)
# =====================================================
EXTRA_PATH = (Path("dataset") / "extra_places.csv") if Path("dataset").is_dir() else Path("extra_places.csv")
EXTRA_COLUMNS = [
    "Name", "City", "State", "Type", "Significance", "Entrance Fee in INR", "Image",
    "Opening Time", "Closing Time", "Weekly Off", "Best Time to visit", "Latitude", "Longitude",
    "Google review rating", "Number of google review in lakhs",
]
EXTRA_SYNONYMS = {
    "Name": ["Name", "Place", "Destination"], "City": ["City", "District"], "State": ["State", "Region"],
    "Type": ["Type", "Category"], "Significance": ["Significance", "Description"],
    "Entrance Fee in INR": ["Entrance Fee in INR", "Fee"], "Image": ["Image", "Image URL"],
    "Opening Time": ["Opening Time", "Timings"], "Closing Time": ["Closing Time"],
    "Weekly Off": ["Weekly Off"], "Best Time to visit": ["Best Time to visit"],
    "Latitude": ["Latitude", "Lat"], "Longitude": ["Longitude", "Lon"],
    "Google review rating": ["Google review rating", "Google Rating", "Rating"],
    "Number of google review in lakhs": ["Number of google review in lakhs", "Google Reviews", "Review Count", "Number of Reviews"],
}
def _osm_contact():
    """Contact e-mail for the OpenStreetMap User-Agent (set OSM_CONTACT_EMAIL env var or Streamlit secret)."""
    val = os.environ.get("OSM_CONTACT_EMAIL", "").strip()
    if val:
        return val
    try:
        return str(st.secrets.get("OSM_CONTACT_EMAIL", "")).strip()
    except Exception:
        return ""


_OSM_CONTACT = _osm_contact()
OSM_HEADERS = {"User-Agent": f"AISmartTouristPlanner/{APP_VERSION} (academic project" + (f"; contact: {_OSM_CONTACT}" if _OSM_CONTACT else "") + ")"}
GENERIC_NAME_WORDS = {
    "temple", "beach", "fort", "palace", "park", "the", "of", "and", "mandir", "sanctuary", "national",
    "museum", "garden", "lake", "falls", "waterfall", "church", "mosque", "sea", "island",
}
TRANSPORT_ORDER = ["Railway station", "Metro", "Bus", "Airport"]
TRANSPORT_ICONS = {"Railway station": "🚉", "Metro": "🚇", "Bus": "🚌", "Airport": "✈️"}


def load_extra_places():
    if EXTRA_PATH.exists():
        try:
            return pd.read_csv(EXTRA_PATH)
        except Exception:
            return None
    return None


def align_extra_to_main(raw, extra):
    """Rename user-added columns to whatever names the main dataset uses."""
    rename = {}
    for canonical, synonyms in EXTRA_SYNONYMS.items():
        target = find_column(raw, synonyms)
        if target is not None and canonical in extra.columns:
            rename[canonical] = target
    return extra.rename(columns=rename)


def save_extra_places(frame):
    EXTRA_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = frame.reindex(columns=EXTRA_COLUMNS)
    out.drop_duplicates(subset=["Name", "City", "State"], keep="last").to_csv(EXTRA_PATH, index=False)
    st.session_state["extra_version"] = st.session_state.get("extra_version", 0) + 1


def append_extra_places(new_rows):
    current = load_extra_places()
    new_df = pd.DataFrame(new_rows).reindex(columns=EXTRA_COLUMNS)
    merged = new_df if current is None or current.empty else pd.concat([current.reindex(columns=EXTRA_COLUMNS), new_df], ignore_index=True)
    save_extra_places(merged)


@st.cache_data(ttl=86400, show_spinner=False)
def geocode_cached(name, city, state):
    return geocode_destination(name, city, state)


@st.cache_data(ttl=7 * 86400, show_spinner=False)
def _wiki_lookup(name, city):
    """Cached Wikipedia thumbnail lookup. Raises RuntimeError on network failure so failures are not cached."""
    tokens = [w for w in re.findall(r"[a-z]+", str(name).lower()) if w not in GENERIC_NAME_WORDS and len(w) > 2]
    if not tokens:
        return None
    try:
        r = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "generator": "search", "gsrsearch": f"{name} {city}".strip(), "gsrlimit": 4,
                "prop": "pageimages", "piprop": "thumbnail", "pithumbsize": 700, "format": "json",
            },
            headers=OSM_HEADERS, timeout=4,
        )
        r.raise_for_status()
        pages = (r.json().get("query") or {}).get("pages") or {}
    except Exception as exc:
        raise RuntimeError("wikipedia unreachable") from exc
    for p in sorted(pages.values(), key=lambda p: p.get("index", 99)):
        title = str(p.get("title", "")).lower()
        thumb = (p.get("thumbnail") or {}).get("source")
        if thumb and all(tok in title for tok in tokens):
            return thumb, p["title"]
    return None


def fetch_wiki_image(name, city=""):
    """(url, page title) or None. Accepted only if the page title contains every distinctive word of the place name."""
    if st.session_state.get("wiki_down"):
        return None
    try:
        return _wiki_lookup(str(name), str(city))
    except RuntimeError:
        st.session_state["wiki_down"] = True
        return None

@st.cache_data(ttl=86400, show_spinner=False)
def download_place_image(url):
    """Download an image and verify that the response contains image data."""
    if not isinstance(url, str) or not url.strip().startswith("http"):
        return None

    try:
        response = requests.get(
            url.strip(),
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120 Safari/537.36"
            },
            timeout=12
        )
        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type", ""
        ).split(";")[0].lower()

        if not content_type.startswith("image/"):
            return None

        image_bytes = response.content

        if not image_bytes or len(image_bytes) > 8 * 1024 * 1024:
            return None

        return image_bytes, content_type

    except Exception:
        return None


def prepare_place_image(image_bytes, size=(640, 360)):
    """Convert destination photos to a consistent card-friendly JPEG size.

    ImageOps.fit keeps the aspect ratio and center-crops excess edges, so images
    fit the UI box without stretching or distortion.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as source:
            source = ImageOps.exif_transpose(source).convert("RGB")
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            fitted = ImageOps.fit(
                source,
                size,
                method=resampling,
                centering=(0.5, 0.45),
            )
            output = io.BytesIO()
            fitted.save(output, format="JPEG", quality=88, optimize=True)
            return output.getvalue()
    except Exception:
        # Keep the original image if it is an unusual format Pillow cannot process.
        return image_bytes


def get_place_image(row):
    """Return a resized destination image, with a browser-side URL fallback."""

    raw_url = row.get("_image", "")
    image_url = str(raw_url).strip() if pd.notna(raw_url) else ""

    # Ignore dataset placeholders such as "No place-specific image".
    if image_url.lower() in {"", "nan", "none", "no place-specific image", "not available", "n/a"}:
        image_url = ""

    # 1. Prefer downloading and converting the dataset image.
    if image_url.lower().startswith(("http://", "https://")):
        result = download_place_image(image_url)
        if result:
            fitted_bytes = prepare_place_image(result[0])
            return fitted_bytes, "image/jpeg", "Dataset image"

        # Some image hosts block Python requests but allow the browser to load
        # the same public URL. Let Streamlit/browser try it directly.
        return image_url, "url", "Dataset image"

    # 2. Try a destination-specific Wikipedia image when enabled.
    if st.session_state.get("wiki_images", True):
        name = str(row.get("_name", "")).strip()
        city = str(row.get("_city", "")).strip()
        wiki = fetch_wiki_image(name, city)

        if wiki and wiki[0]:
            result = download_place_image(wiki[0])
            if result:
                fitted_bytes = prepare_place_image(result[0])
                return fitted_bytes, "image/jpeg", f"Wikipedia: {wiki[1]}"

            # Browser-side fallback for Wikimedia hotlink/request restrictions.
            return wiki[0], "url", f"Wikipedia: {wiki[1]}"

    return None
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_mappls_transport(lat, lon, radius_km, api_key):
    """Fetch nearby transport POIs from Mappls Nearby API using a Static Key."""
    if not api_key:
        raise RuntimeError("Mappls Static Key is not configured.")

    endpoint = "https://search.mappls.com/search/places/nearby/json"
    radius_m = max(500, min(int(float(radius_km) * 1000), 10000))
    searches = [
        ("Railway station", "railway station"),
        ("Metro", "metro station"),
        ("Bus", "bus station"),
        ("Bus", "bus stop"),
        ("Airport", "airport"),
    ]
    rows = []
    errors = []

    for category, keyword in searches:
        try:
            response = requests.get(
                endpoint,
                params={
                    "keywords": keyword,
                    "refLocation": f"{lat},{lon}",
                    "radius": radius_m,
                    "region": "IND",
                    "sortBy": "dist:asc",
                    "page": 1,
                    "access_token": api_key,
                },
                headers={"User-Agent": f"AISmartTouristPlanner/{APP_VERSION}"},
                timeout=(5, 18),
            )
            if response.status_code in (401, 403):
                raise RuntimeError("Mappls rejected the key or this API is not enabled for the application.")
            response.raise_for_status()
            payload = response.json()
            locations = payload.get("suggestedLocations", []) if isinstance(payload, dict) else []
            for item in locations:
                name = str(item.get("placeName") or item.get("name") or "").strip()
                if not name:
                    continue
                codes = [str(c).upper() for c in (item.get("keywords") or [])]
                row_category = category
                if codes == ["TRNBUS"]:
                    if category in ("Railway station", "Metro", "Airport"):
                        continue  # Mappls returned a bus stop for a rail/metro/airport search
                    row_category = "Bus"
                try:
                    distance_km = float(item.get("distance", 0)) / 1000.0
                except (TypeError, ValueError):
                    distance_km = None
                rows.append({
                    "Category": row_category,
                    "Name": name,
                    "Distance (km)": round(distance_km, 2) if distance_km is not None else None,
                    "Detail": str(item.get("placeAddress") or item.get("type") or "").strip(),
                    "eLoc": str(item.get("eLoc") or "").strip(),
                    "Source": "Mappls Nearby API",
                })
        except RuntimeError:
            raise
        except (requests.RequestException, ValueError) as exc:
            errors.append(f"{category}: {type(exc).__name__}")

    result = pd.DataFrame(rows, columns=[
        "Category", "Name", "Distance (km)", "Detail", "eLoc", "Source"
    ])
    if not result.empty:
        # Many different stops are all called "Bus Stop", so de-duplicate by Mappls id (eLoc), not by name.
        has_id = result["eLoc"].astype(str).str.len() > 0
        with_id = result[has_id].drop_duplicates(subset=["eLoc"])
        no_id = result[~has_id].drop_duplicates(subset=["Category", "Name"])
        result = pd.concat([with_id, no_id], ignore_index=True)
        # Collapse "Airport Arrival Terminal 2", "Departure Terminal 2", "Old Terminal" etc. into one airport.
        is_air = result["Category"] == "Airport"
        if is_air.any():
            air = result[is_air].copy()
            air["Name"] = air["Name"].str.replace(
                r"\s*[-,]?\s*((arrival|departure|old|new|domestic|international|cargo)\s+)*terminal\s*[\w-]*\s*$|\s+(arrival|departure)s?\s*$",
                "", regex=True, flags=re.IGNORECASE).str.strip()
            air = air.sort_values("Distance (km)", na_position="last").drop_duplicates(subset=["Name"])
            result = pd.concat([result[~is_air], air], ignore_index=True)
        result = result.sort_values("Distance (km)", na_position="last").reset_index(drop=True)
    if result.empty and errors:
        raise RuntimeError("Mappls could not complete the nearby searches. Please check the key, API access and connection.")
    return result


def get_mappls_static_key():
    """Read the Mappls key from Streamlit secrets or the MAPPLS_API_KEY environment variable."""
    key = os.environ.get("MAPPLS_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets.get("MAPPLS_API_KEY", "")).strip()
    except Exception:
        return ""


def _overpass(query):
    """Fetch OSM features with short, bounded retries so the page cannot hang for minutes."""
    endpoints = (
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    )
    last_error = None
    for url in endpoints:
        try:
            response = requests.post(
                url,
                data={"data": query},
                headers=OSM_HEADERS,
                timeout=(4, 12),  # connect timeout, response timeout
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or "elements" not in payload:
                raise ValueError("Unexpected response from map service")
            return payload.get("elements", [])
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            continue
    return None

def _classify_transport(tags):
    """Classify OSM transport features, including common Indian metro tagging styles."""
    tags = tags or {}
    name = str(tags.get("name:en") or tags.get("name") or "").lower()
    network = " ".join(str(tags.get(k, "")) for k in ("network", "operator", "brand")).lower()
    railway = str(tags.get("railway", "")).lower()
    station = str(tags.get("station", "")).lower()
    public_transport = str(tags.get("public_transport", "")).lower()

    if tags.get("aeroway") == "aerodrome":
        return "Airport"
    if tags.get("amenity") == "bus_station":
        return "Bus"
    if tags.get("highway") == "bus_stop" or tags.get("public_transport") == "platform" and tags.get("bus") == "yes":
        return "Bus"

    # Metro systems can be mapped as subway, light rail, monorail, or generic
    # railway/public-transport stations with a metro operator/network.
    metro_hint = any(x in (network + " " + name) for x in (
        "metro", "subway", "rapid transit", "rapidtransit", "monorail"
    ))
    if (
        railway in ("subway", "light_rail", "monorail")
        or station in ("subway", "light_rail", "monorail", "metro")
        or tags.get("subway") == "yes"
        or tags.get("light_rail") == "yes"
        or tags.get("monorail") == "yes"
        or (public_transport == "station" and metro_hint)
        or (railway in ("station", "halt") and metro_hint)
    ):
        return "Metro"

    if railway in ("station", "halt") or public_transport == "station":
        return "Railway station"
    return None


@st.cache_data(ttl=30 * 60, show_spinner=False)
def fetch_transport(lat, lon, radius_km):
    """Search nearby mapped transport in one bounded Overpass request.

    A short cache avoids repeatedly querying public OSM servers while testing.
    Results are only returned when they are actually present in map data.
    """
    lat, lon = float(lat), float(lon)
    radius_m = max(1000, min(int(float(radius_km) * 1000), 100000))
    bus_stop_radius_m = min(radius_m, 8000)

    # One combined request is faster and avoids waiting on a separate airport query.
    query = (
        f'[out:json][timeout:18];('
        f'nwr(around:{radius_m},{lat},{lon})["railway"~"^(station|halt|subway|light_rail|monorail)$"]["name"];'
        f'nwr(around:{radius_m},{lat},{lon})["station"~"^(subway|light_rail|monorail|metro)$"]["name"];'
        f'nwr(around:{radius_m},{lat},{lon})["subway"="yes"]["name"];'
        f'nwr(around:{radius_m},{lat},{lon})["light_rail"="yes"]["name"];'
        f'nwr(around:{radius_m},{lat},{lon})["public_transport"="station"]["name"];'
        f'nwr(around:{radius_m},{lat},{lon})["amenity"="bus_station"]["name"];'
        f'nwr(around:{bus_stop_radius_m},{lat},{lon})["highway"="bus_stop"]["name"];'
        f'nwr(around:{bus_stop_radius_m},{lat},{lon})["public_transport"="platform"]["bus"="yes"]["name"];'
        f'nwr(around:150000,{lat},{lon})["aeroway"="aerodrome"]["name"];'
        f');out center tags 1200;'
    )

    elements = _overpass(query)
    if elements is None:
        raise RuntimeError("OpenStreetMap servers did not respond within the time limit")

    rows = {}
    for element in elements:
        tags = element.get("tags") or {}
        category = _classify_transport(tags)
        name = str(tags.get("name:en") or tags.get("name") or "").strip()
        if not category or not name:
            continue
        center = element.get("center") or {}
        raw_lat = element.get("lat", center.get("lat"))
        raw_lon = element.get("lon", center.get("lon"))
        if raw_lat is None or raw_lon is None:
            continue
        try:
            point_lat, point_lon = float(raw_lat), float(raw_lon)
            distance = haversine_km((lat, lon), (point_lat, point_lon))
        except (TypeError, ValueError):
            continue
        if not (6 <= point_lat <= 38 and 68 <= point_lon <= 98):
            continue
        if category == "Airport":
            if distance > 150:
                continue
        elif distance > float(radius_km):
            continue

        detail = (
            tags.get("iata") or tags.get("ref") or tags.get("operator")
            or tags.get("network") or tags.get("railway", "")
        )
        key = (category, re.sub(r"\s+", " ", name).casefold())
        if key not in rows or distance < rows[key][0]:
            rows[key] = (distance, point_lat, point_lon, str(detail), name)

    data = [{
        "Category": category,
        "Name": values[4],
        "Distance (km)": round(values[0], 2),
        "Detail": values[3],
        "lat": values[1],
        "lon": values[2],
        "Source": "OpenStreetMap",
    } for (category, _), values in rows.items()]
    result = pd.DataFrame(data, columns=[
        "Category", "Name", "Distance (km)", "Detail", "lat", "lon", "Source"
    ])
    if not result.empty:
        result = result.sort_values("Distance (km)").reset_index(drop=True)
    return result


# =====================================================
# LUXURY STYLING ENGINE & UI COMPONENTS
# =====================================================
st.markdown("""
<style>
:root {
    --olive: #71834A;
    --olive-dark: #354522;
    --olive-light: #87995D;
    --cream: #F2E9D8;
    --ink: #27311F;
    --line: #DCE3D1;
}
.stApp {
    background: var(--cream) !important;
    color: var(--ink) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
[data-testid="stAppViewContainer"] { background: var(--cream) !important; }
[data-testid="stMain"] { background: #F2E9D8 !important; }
[data-testid="stMainBlockContainer"] { background: transparent !important; }
[data-testid="stHeader"], [data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: var(--cream) !important;
    color: var(--olive-dark) !important;
}
[data-testid="stHeader"] button, [data-testid="stToolbar"] button { color: var(--olive-dark) !important; }
.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1500px; }
h1, h2, h3, h4, [data-testid="stMetricLabel"] {
    color: var(--olive-dark) !important;
    font-weight: 700 !important;
}
p, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: #35402D; }
/* Slightly larger, clearer text for the Explore / Place details navigation. */
.stRadio [role="radiogroup"] label {
    font-size: 1.05rem !important;
    font-weight: 650 !important;
}

/* Strong overrides for Streamlit/BaseWeb controls: remove navy-blue fields. */
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
[data-testid="stMultiSelect"] [data-baseweb="select"] > div > div,
[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stMultiSelect"] [role="combobox"] {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    color: #27311F !important;
    border-color: #AAB795 !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stMultiSelect"] [data-baseweb="select"] * {
    color: #27311F !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] svg,
[data-testid="stMultiSelect"] [data-baseweb="select"] svg {
    fill: #354522 !important;
    color: #354522 !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div,
[data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within > div {
    border: 2px solid #71834A !important;
    box-shadow: 0 0 0 1px #71834A !important;
}
/* Explore / Place details navigation: visible, roomy cream-and-olive panel. */
[data-testid="stVerticalBlockBorderWrapper"]:has(.stRadio) {
    background: #EFE3CF !important;
    border: 2px solid #71834A !important;
    border-radius: 14px !important;
    padding: 8px 14px !important;
    margin: 4px 0 18px 0 !important;
}
.stRadio [role="radiogroup"] {
    display: flex !important;
    align-items: center !important;
    gap: 22px !important;
    min-height: 48px !important;
}
.stRadio [role="radiogroup"] label {
    font-size: 1.15rem !important;
    font-weight: 750 !important;
    color: #27311F !important;
    padding: 8px 12px !important;
    border-radius: 9px !important;
}
.stRadio [role="radiogroup"] label p {
    color: #27311F !important;
    font-size: 1.15rem !important;
    font-weight: 750 !important;
}

/* Olive sidebar */
section[data-testid="stSidebar"] {
    background: var(--olive-dark) !important;
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] * { color: #FFFFFF; }
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    color: #27311F !important;
    background: #FFFFFF !important;
}

/* White input fields and dropdowns */
.stTextInput input, .stNumberInput input, .stDateInput input,
.stTimeInput input, .stTextArea textarea,
[data-baseweb="select"] > div,
[data-baseweb="select"] [role="combobox"],
[data-baseweb="input"] > div,
[data-testid="stFileUploader"] section {
    background: #FFFFFF !important;
    color: #27311F !important;
    border: 1px solid #C9D3BC !important;
    border-radius: 10px !important;
    box-shadow: none !important;
}
/* Keep dropdowns and multiselects in the planner's olive-and-cream theme. */
div[data-baseweb="select"] *,
div[data-baseweb="popover"] *,
[data-baseweb="menu"] * {
    color: #27311F !important;
}
div[data-baseweb="popover"],
div[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    background: #FFFFFF !important;
    border: 1px solid #DCE3D1 !important;
    border-radius: 10px !important;
}
[data-baseweb="select"] input { color: #27311F !important; }
[data-baseweb="select"] svg { fill: var(--olive-dark) !important; }
[data-baseweb="tag"] {
    background: #E8EDDF !important;
    border: 1px solid #C9D3BC !important;
    border-radius: 7px !important;
}
[data-baseweb="menu"] [aria-selected="true"],
[role="option"][aria-selected="true"] {
    background: #E8EDDF !important;
}
[data-baseweb="select"]:focus-within > div,
[data-baseweb="input"]:focus-within > div {
    border-color: var(--olive) !important;
    box-shadow: 0 0 0 1px var(--olive) !important;
}
[data-testid="stFileUploader"] section {
    background: #FFFFFF !important;
    border: 1px dashed #9AAA7A !important;
}
[data-testid="stFileUploader"] button {
    background: var(--olive) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--olive) !important;
}
.stSlider [data-baseweb="slider"] div[role="slider"] { background: var(--olive) !important; }

/* Cards, buttons and tabs */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #FFFFFF !important;
    border: 1px solid var(--line) !important;
    border-radius: 16px !important;
    box-shadow: 0 4px 16px rgba(53, 69, 34, 0.07);
}
.stButton > button, .stDownloadButton > button {
    background: var(--olive) !important;
    color: #FFFFFF !important;
    border: 1px solid var(--olive) !important;
    border-radius: 10px !important;
    font-weight: 650 !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    background: var(--olive-dark) !important;
    border-color: var(--olive-dark) !important;
    color: #FFFFFF !important;
}
/* Main navigation: large, high-contrast feature tiles */
.stTabs [data-baseweb="tab-list"] {
    display: grid !important;
    grid-template-columns: repeat(3, minmax(0, 1fr)) !important;
    gap: 12px !important;
    width: 100% !important;
    overflow: visible !important;
    white-space: normal !important;
    background: transparent !important;
    padding: 8px 2px 16px !important;
    border: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    min-width: 0 !important;
    min-height: 60px !important;
    height: auto !important;
    padding: 13px 12px !important;
    white-space: normal !important;
    text-align: center !important;
    color: #27311F !important;
    background: #FFFFFF !important;
    border: 2px solid #354522 !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 7px rgba(39, 49, 31, 0.08) !important;
}
.stTabs [data-baseweb="tab"] p {
    color: #27311F !important;
    font-size: 17px !important;
    line-height: 1.3 !important;
    font-weight: 750 !important;
    margin: 0 !important;
    white-space: normal !important;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #F0F3E9 !important;
    border: 2px solid #273719 !important;
    transform: translateY(-1px);
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    color: #27311F !important;
    background: #E9EEDF !important;
    border: 3px solid #273719 !important;
    box-shadow: inset 0 0 0 1px #FFFFFF !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] p {
    color: #27311F !important;
    font-weight: 800 !important;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}
@media (max-width: 900px) {
    .stTabs [data-baseweb="tab-list"] {
        grid-template-columns: repeat(2, minmax(0, 1fr)) !important;
    }
    .stTabs [data-baseweb="tab"] p { font-size: 16px !important; }
}
@media (max-width: 560px) {
    .stTabs [data-baseweb="tab-list"] {
        grid-template-columns: 1fr !important;
    }
}
[data-testid="stMetric"] {
    background: #FFFFFF; border: 1px solid var(--line);
    border-radius: 14px; padding: 14px 16px;
}
[data-testid="stDataFrame"], [data-testid="stTable"] { background: #FFFFFF; }
/* Keep destination thumbnails visually consistent across cards and recommendation results. */
div[data-testid="stImage"] img { width: 100% !important; height: auto !important; max-height: 220px !important; object-fit: contain !important; background: #F4F1E8; border-radius: 12px !important; }
.stAlert { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,600;9..144,700&display=swap');
:root { --accent: #C2603A; --accent-dark: #9E4A2A; --accent-soft: #FBEBE3; --serif: 'Fraunces', Georgia, 'Times New Roman', serif; }
h1, h2, h3, .home-hero h1, .pcard-title, .home-feature h4, .home-step h4 { font-family: var(--serif) !important; }
.stButton > button[kind="primary"], button[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primaryFormSubmit"], .stFormSubmitButton > button {
    background: var(--accent) !important; border-color: var(--accent) !important; color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover,
button[data-testid="stBaseButton-primaryFormSubmit"]:hover, .stFormSubmitButton > button:hover {
    background: var(--accent-dark) !important; border-color: var(--accent-dark) !important;
}
.hero-tag { background: var(--accent) !important; border-color: var(--accent) !important; }
.home-feature:hover, .pcard:hover { border-color: var(--accent) !important; }
</style>
""", unsafe_allow_html=True)


def render_hero_banner(state_name, place_count):
    focus_text = (
        f"Currently exploring destinations in {html.escape(str(state_name))}."
        if state_name != "All India"
        else "Discover amazing destinations across India."
    )
    st.markdown("""
    <style>
    .home-hero {
        min-height: 270px;
        padding: 40px 42px;
        border-radius: 22px;
        overflow: hidden;
        background:
            linear-gradient(90deg, rgba(20, 35, 12, .88), rgba(20, 35, 12, .34)),
            url('https://images.unsplash.com/photo-1506461883276-594a12b11cf3?auto=format&fit=crop&w=1800&q=85');
        background-size: cover;
        background-position: center;
        border: 1px solid rgba(102, 122, 62, .35);
        box-shadow: 0 10px 28px rgba(35, 48, 22, .14);
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .hero-tag {
        display: inline-block;
        width: fit-content;
        background: rgba(255,255,255,.16);
        color: #FFFFFF;
        border: 1px solid rgba(255,255,255,.55);
        border-radius: 30px;
        padding: 7px 14px;
        font-size: 13px;
        font-weight: 700;
        margin-bottom: 15px;
    }
    .home-hero h1 {
        color: #FFFFFF !important;
        font-size: clamp(2.2rem, 5vw, 3.5rem) !important;
        margin: 0 0 12px 0 !important;
        line-height: 1.12;
    }
    .home-hero p {
        color: #F4F6EF !important;
        font-size: 17px;
        max-width: 650px;
        line-height: 1.65;
        margin: 0;
    }
    .home-focus { color: #E8F0D7; font-size: 13px; margin-top: 14px; }
    .home-feature {
        background: #FFFCF6;
        border: 1px solid #D7C9B4;
        border-radius: 16px;
        padding: 23px 20px;
        min-height: 170px;
        height: 100%;
        box-sizing: border-box;
        box-shadow: 0 4px 14px rgba(53,67,33,.06);
        transition: .2s ease;
    }
    .home-feature:hover { border-color: #87995D; transform: translateY(-3px); }
    .home-feature .feature-icon { font-size: 28px; margin-bottom: 10px; }
    .home-feature h4 { color: #354321 !important; font-size: 17px; margin: 0 0 8px; }
    .home-feature p { color: #59634D !important; font-size: 14px; line-height: 1.55; margin: 0; }
    div[data-testid="stImage"] img { width: 100% !important; height: auto !important; max-height: 220px !important; object-fit: contain !important; background: #F4F1E8; border-radius: 12px !important; }
    .pcard { min-height: 245px; display:flex; flex-direction:column; }
    .pcard-body { flex:1; }
    .pcard-title { min-height: 2.5em; }
    .pcard-desc { min-height: 4.5em; }
    .home-step {
        background: #EEF1E7;
        border: 1px solid #DCE3D1;
        border-radius: 14px;
        padding: 18px;
        min-height: 125px;
    }
    .home-step span { color: #667A3E; font-size: 23px; font-weight: 800; }
    .home-step h4 { color: #354321 !important; font-size: 15px; margin: 8px 0 5px; }
    .home-step p { color: #59634D !important; font-size: 13px; margin: 0; }
    @media(max-width: 700px) { .home-hero { padding: 30px 22px; min-height: 290px; } }
    </style>
    <div class="home-hero">
        <div class="hero-tag">✦ AI POWERED TRAVEL PLANNING</div>
        <h1>Your Journey,<br> Your Way.</h1>
        <p>Discover beautiful destinations, get personalized AI recommendations and plan your perfect trip with ease.</p>
        <div class="home-focus">🌿 Explore India&nbsp; • &nbsp;Plan Smarter&nbsp; • &nbsp;Travel Better</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f'<p class="home-focus">{focus_text}</p>', unsafe_allow_html=True)

def render_metric_card(label, value, icon):
    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #FFFFFF 0%, #F0F3E9 100%);
        border: 1px solid #DCE3D1; border-radius: 14px; padding: 18px; text-align: center;
    ">
        <div style="font-size: 1.8rem; margin-bottom: 4px;">{icon}</div>
        <div style="color: #59634D; font-size: 0.85rem; font-weight: 500;">{label}</div>
        <div style="color: #354522; font-size: 1.6rem; font-weight: 700; margin-top: 2px;">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def render_dataset_transport(row):
    """Nearest metro / railway connection stored in the dataset (works without internet or API keys)."""
    tr = str(row.get("_nearest_transport", "") or "").strip()
    if not tr:
        return
    v = row.get("_transport_verified")
    verified = bool(v) if isinstance(v, (bool, np.bool_)) else False
    st.markdown(
        f"🚉 **Nearest connection (from the dataset):** {tr}"
        + ("" if verified else "  \n_Generic estimate, not a verified station._")
    )
def show_destination(row, score=None, tab_name="gen"):

    safe_name = str(row["_name"]).replace(" ", "_").replace(".", "")
    safe_city = str(row.get("_city", "")).replace(" ", "_").replace(".", "")
    idx_str = str(getattr(row, "name", "0"))

    with st.container(border=True):

        # Load image once
        image_info = get_place_image(row)

        city_tag = f"📍 {row.get('_city')}" if row.get("_city") else ""
        type_tag = f"🏷️ {row.get('_type')}" if row.get("_type") else ""

        fee = row.get("_fee")

        fee_text = (
            "Not listed"
            if pd.isna(fee)
            else ("Free" if fee == 0 else f"₹{fee:,.0f}")
        )

        opening = row.get("_opening_time", "")
        closing = row.get("_closing_time", "")

        opening = "" if pd.isna(opening) else str(opening).strip()
        closing = "" if pd.isna(closing) else str(closing).strip()

        timing_text = (
            f"{opening} - {closing}"
            if opening and closing
            else "Not available"
        )

        # Common destination information
        def show_place_information():

            st.markdown(f"### {row['_name']}")

            st.caption(
                f"{city_tag} | {type_tag} | "
                f"State: {row.get('_state', 'N/A')}"
            )

            description = row.get("_description", "")

            if pd.notna(description) and str(description).strip():
                st.write(str(description)[:320])

            c_a, c_b = st.columns(2)

            c_a.markdown(f"💵 **Entry Fee:** `{fee_text}`")
            c_b.markdown(f"⏰ **Timings:** `{timing_text}`")

        # Show image and information together
        if image_info:

            left, right = st.columns([1, 2.5])

            with left:

                st.image(
                    image_info[0],
                    use_container_width=True
                )

                if image_info[2]:
                    st.caption(image_info[2])

            with right:
                show_place_information()

        else:

            st.info("📷 Image not available for this destination.")

            show_place_information()

        render_dataset_transport(row)

        st.divider()

        action_col, score_col = st.columns([2, 1])

        place_id = f"{row['_name']}_{row.get('_city', '')}"

        button_key = f"{tab_name}_{safe_name}_{safe_city}_{idx_str}"

        with action_col:

            if place_id in st.session_state["favorites"]:

                if st.button(
                    "❤️ Saved in Favorites",
                    key=f"rem_{button_key}"
                ):
                    st.session_state["favorites"].remove(place_id)
                    save_favorites()
                    st.rerun()

            else:

                if st.button(
                    "🤍 Add to Favorites",
                    key=f"add_{button_key}"
                ):
                    st.session_state["favorites"].append(place_id)
                    save_favorites()
                    st.rerun()
        with score_col:

            if score is not None:
                st.markdown(
                    f"🎯 **Match Score:** `{score:.1f}%`"
                )
# =====================================================
# DESTINATION CARDS (image-first grid)
# =====================================================
st.markdown("""
<style>
.pcard { background:#fff; border:1px solid #DCE3D1; border-radius:16px; overflow:hidden;
         box-shadow:0 4px 14px rgba(53,69,34,.07); transition:.2s ease; margin-bottom:6px; }
.pcard:hover { transform:translateY(-3px); border-color:#87995D; }
.pcard-img { height:180px; background-size:cover; background-position:center; position:relative; }
.pcard-type { position:absolute; left:10px; top:10px; background:rgba(255,255,255,.92); color:#27311F;
              border-radius:20px; padding:4px 11px; font-size:12px; font-weight:700; }
.pcard-body { padding:14px 16px 12px; }
.pcard-title { color:#354522; font-size:17px; font-weight:750; line-height:1.25; min-height:2.5em; }
.pcard-loc { color:#59634D; font-size:13px; margin:4px 0 8px; }
.pcard-badges { display:flex; flex-wrap:wrap; gap:6px; margin-bottom:8px; min-height:26px; }
.pc-badge { background:#EEF1E7; color:#354522; border-radius:20px; padding:3px 10px; font-size:12px; font-weight:650; }
.pc-badge.free { background:#DDEFD0; color:#245016; }
.pc-badge.muted { background:#F1F1EC; color:#6B7160; }
.pc-badge.season { background:#FBEBE3; color:#9E4A2A; }
.pcard-rate { position:absolute; right:10px; top:10px; background:rgba(39,49,31,.82); color:#fff;
              border-radius:20px; padding:4px 10px; font-size:12px; font-weight:700; }
.pcard-meta { color:#59634D; font-size:12px; margin-bottom:6px; }
.pcard-desc { color:#59634D; font-size:13px; line-height:1.5; min-height:4.5em; }
</style>
""", unsafe_allow_html=True)

TYPE_ICONS = [
    ("beach", "🏖️"), ("temple", "🛕"), ("fort", "🏰"), ("palace", "🏯"), ("museum", "🏛️"),
    ("park", "🌳"), ("garden", "🌷"), ("wildlife", "🐯"), ("lake", "🏞️"), ("hill", "⛰️"),
    ("waterfall", "💧"), ("church", "⛪"), ("mosque", "🕌"), ("monument", "🗿"),
]
HOME_CHIPS = [
    ("🏖️ Beaches", "beach"), ("🏞️ Lakes", "lake"), ("🏰 Forts & palaces", "fort"),
    ("🛕 Temples", "temple"), ("🌳 Parks & nature", "park"), ("⛰️ Mountains", "mountain"),
]


def type_icon(type_text):
    t = str(type_text).lower()
    for key, icon in TYPE_ICONS:
        if key in t:
            return icon
    return "📍"


def toggle_favorite(place_id):
    favs = st.session_state["favorites"]
    if place_id in favs:
        favs.remove(place_id)
    else:
        favs.append(place_id)
    save_favorites()


def _open_details(name):
    if name in active_df["_name"].values:
        st.session_state["detail_select"] = name
        st.session_state["sec_explore"] = EXPLORE_SECTIONS[1]
def render_place_card(row, key_prefix, idx=0, image_width=None):

    """Display destination cards with reliable Streamlit image rendering."""

    esc = lambda v: html.escape(str(v if v is not None else ""))

    name = str(row["_name"])

    loc = ", ".join(
        x for x in [
            str(row.get("_city", "") or ""),
            str(row.get("_state", "") or "")
        ] if x
    )

    # Load destination image
    image_info = get_place_image(row)

    # Display actual image using Streamlit
    if image_info:

        if image_width:
            st.image(image_info[0], width=image_width)
        else:
            st.image(image_info[0], use_container_width=True)

    else:

        st.markdown(
            """
            <div style="
                height:150px;
                background:linear-gradient(135deg,#87995D,#354522);
                border-radius:12px;
                display:flex;
                align-items:center;
                justify-content:center;
                color:white;
                font-size:45px;
            ">📍</div>
            """,
            unsafe_allow_html=True
        )

    # Entry fee
    badges = []

    fee = row.get("_fee")

    if pd.isna(fee):
        badges.append(("muted", "Fee not listed"))

    elif fee == 0:
        badges.append(("free", "Free entry"))

    else:
        badges.append(("", f"₹{fee:,.0f}"))

    # Rating
    rating = row.get("_rating")

    rate_html = (
        f"⭐ {float(rating):.1f}"
        if pd.notna(rating)
        else "Rating not listed"
    )

    # Opening hours
    open_t = str(row.get("_opening_time", "") or "")
    close_t = str(row.get("_closing_time", "") or "")

    hours_txt = (
        f"{open_t} - {close_t}"
        if open_t and close_t
        else "Hours not listed"
    )

    # Season
    best = row.get("_best_time", "")

    if (
        str(best).strip().lower() not in ("", "all", "nan")
        and season_fit_place(
            best,
            row.get("_type", ""),
            datetime.date.today().month - 1
        ) == 1.0
    ):
        badges.append(("season", "In season now"))

    badge_html = "".join(
        f'<span class="pc-badge {cls}">{esc(txt)}</span>'
        for cls, txt in badges
    )

    # Description
    desc = str(row.get("_description", "") or "")

    desc = (
        desc[:110].rstrip() + "…"
        if len(desc) > 110
        else desc
    )
        # Card information
    card_html = textwrap.dedent(f"""
    <div class="pcard">
        <div class="pcard-body">

            <div class="pcard-title">
                {esc(name)}
            </div>

            <div class="pcard-loc">
                📍 {esc(loc) if loc else "Location not listed"}
            </div>

            <div class="pcard-badges">
                {badge_html}
            </div>

            <div class="pcard-meta">
                {rate_html} | 🕘 {esc(hours_txt)}
            </div>

            <div class="pcard-desc">
                {esc(desc)}
            </div>

        </div>
    </div>
    """).strip()

        # Render destination card as HTML directly
    st.html(card_html)
    # Buttons
    place_id = f"{name}_{row.get('_city', '')}"

    saved = place_id in st.session_state["favorites"]

    b1, b2 = st.columns(2)

    b1.button(
        "❤️ Saved" if saved else "🤍 Save",
        key=f"cardfav_{key_prefix}_{idx}",
        on_click=toggle_favorite,
        args=(place_id,),
        use_container_width=True
    )

    if b2.button(
        "Details",
        key=f"carddet_{key_prefix}_{idx}",
        use_container_width=True,
        on_click=_open_details,
        args=(name,)
    ):

        if (
            name in active_df["_name"].values
            and current_page.title != PAGES["explore"].title
        ):
            st.switch_page(PAGES["explore"])
# =====================================================
# ACCOUNT, PROFILE & MOBILE LAYOUT (v2.4)
# =====================================================
st.markdown("""
<style>
/* Sidebar-free premium navigation */
.brand-mark{padding:7px 0;color:#354522;font-size:21px;white-space:nowrap}
.brand-mark small{display:block;font-size:9px;letter-spacing:2px;color:#8a927d;margin-left:30px;margin-top:-3px}
.top-nav-rule{height:1px;background:#e2e6da;margin:4px 0 18px}
.profile-pop{text-align:center;padding:8px 4px 14px}
.profile-avatar{width:66px;height:66px;border-radius:50%;background:#f4eee4;display:flex;align-items:center;justify-content:center;font-size:34px;margin:auto;border:2px solid #dce3d1}
.profile-pop h3{margin:8px 0 0;color:#354522}.profile-pop p{margin:0;color:#77806d;font-size:12px}
.home-feature{height:160px;border:1px solid #dce3d1;border-radius:17px;padding:18px 16px;background:linear-gradient(145deg,#fffdf8,#f3f5ed);box-shadow:0 4px 14px rgba(53,69,34,.055);margin-bottom:9px;transition:transform .15s,box-shadow .15s}
.home-feature:hover{transform:translateY(-3px);box-shadow:0 9px 22px rgba(53,69,34,.11)}
.feature-icon{font-size:29px;margin-bottom:9px}.home-feature h4{font-size:16px;color:#354522;margin:0 0 6px}.home-feature p{font-size:12.5px;line-height:1.45;color:#68715e;margin:0}
@media(max-width:850px){.brand-mark{font-size:16px}.brand-mark small{font-size:7px;margin-left:24px}.home-feature{height:175px;padding:13px 11px}.home-feature h4{font-size:14px}.home-feature p{font-size:11.5px}}
/* Welcome "boarding pass" */
.pass{display:flex;align-items:stretch;background:#fff;border:1px solid #DCE3D1;border-radius:18px;
      box-shadow:0 4px 16px rgba(53,69,34,.08);margin:0 0 16px;overflow:hidden}
.pass-main{flex:1.6;display:flex;align-items:center;gap:14px;padding:16px 20px;min-width:0}
.pass-av{width:46px;height:46px;border-radius:50%;background:#FBEBE3;display:flex;align-items:center;justify-content:center;font-size:24px;flex:none}
.pass-hi{font-family:var(--serif);font-size:1.4rem;font-weight:700;color:#354522;line-height:1.2}
.pass-sub{font-size:13px;color:#59634D;margin-top:2px}
.pass-stub{flex:1;padding:14px 18px;border-left:2px dashed #C9D3BC;position:relative;display:flex;flex-direction:column;justify-content:center;min-width:0}
.pass-stub::before,.pass-stub::after{content:"";position:absolute;left:-9px;width:16px;height:16px;border-radius:50%;background:var(--cream);border:1px solid #DCE3D1}
.pass-stub::before{top:-9px}.pass-stub::after{bottom:-9px}
.pass-k{font-size:12px;color:#59634D}
.pass-v{font-size:15px;font-weight:700;color:#9E4A2A;margin-top:2px}
.side-user{display:flex;align-items:center;gap:10px;padding:10px 12px;margin:4px 0 8px;border-radius:12px;background:rgba(255,255,255,.12)}
.side-user .av{font-size:24px}.side-user b{display:block;font-size:15px}.side-user small{opacity:.8}
.statebar-note{font-size:12.5px;color:#59634D;margin:-6px 0 8px}
.trip-chip{background:#EEF1E7;border:1px solid #DCE3D1;border-radius:12px;padding:9px 12px;font-size:13.5px;color:#354522;margin-top:1.7rem}
@media(max-width:700px){
  .pass{flex-direction:column}
  .pass-stub{border-left:0;border-top:2px dashed #C9D3BC}
  .pass-stub::before,.pass-stub::after{top:-9px;bottom:auto}
  .pass-stub::before{left:-9px}.pass-stub::after{left:auto;right:-9px}
  .trip-chip{margin-top:0}
}
/* Month navigation stays on one row, even on phones */
.st-key-cal_nav [data-testid="stHorizontalBlock"]{flex-direction:row!important;flex-wrap:nowrap!important;gap:8px!important}
.st-key-cal_nav [data-testid="stColumn"],.st-key-cal_nav [data-testid="column"]{min-width:0!important;flex:1 1 0!important}
/* Bottom tab bar (phones only) */
.st-key-mobnav{display:none}
@media(max-width:768px){
  .block-container{padding:1rem .8rem 6.5rem!important}
  h1{font-size:1.6rem!important}h2{font-size:1.35rem!important}h3{font-size:1.15rem!important}
  .home-hero{min-height:220px!important;padding:24px 18px!important}
  .stRadio [role="radiogroup"]{flex-wrap:wrap!important;gap:6px!important;min-height:0!important}
  .stRadio [role="radiogroup"] label,.stRadio [role="radiogroup"] label p{font-size:1rem!important;padding:6px 8px!important}
  input,textarea{font-size:16px!important}
  .stButton>button,.stDownloadButton>button,.stFormSubmitButton>button{min-height:46px}
  .st-key-chat_fab{right:14px!important;bottom:86px!important}
  .st-key-chat_fab button{width:54px!important;height:54px!important;min-height:54px!important}
  div[data-testid="stPopoverBody"]{min-width:0!important;width:94vw!important}
  .st-key-mobnav{display:block;position:fixed;left:0;right:0;bottom:0;z-index:999;background:#fff;border-top:1px solid #DCE3D1;
     padding:4px 4px calc(4px + env(safe-area-inset-bottom));box-shadow:0 -4px 16px rgba(53,69,34,.10)}
  .st-key-mobnav [data-testid="stHorizontalBlock"]{flex-direction:row!important;flex-wrap:nowrap!important;gap:0!important}
  .st-key-mobnav [data-testid="stColumn"],.st-key-mobnav [data-testid="column"]{min-width:0!important;flex:1 1 0!important;width:auto!important}
  .st-key-mobnav a{flex-direction:column!important;gap:0!important;padding:4px 0!important;justify-content:center;text-align:center}
  .st-key-mobnav a p{font-size:10.5px!important;line-height:1.1!important;margin:0!important}
  .st-key-mobnav a [data-testid="stIconEmoji"]{font-size:20px!important}
}
</style>
""", unsafe_allow_html=True)


def profile_save():
    user, prof = st.session_state.get("user"), st.session_state.get("profile")
    if prof is not None and user and not user.get("guest"):
        save_profile(user["username"], prof)


def _login_user(username, prof, guest=False):
    st.session_state["user"] = {"username": username, "guest": guest}
    st.session_state["profile"] = prof
    st.session_state["favorites"] = list(prof.get("favorites", []))
    st.session_state.setdefault("pack_custom", list(prof.get("packing", {}).get("custom", [])))
    if prof.get("home_city") and "ai_start_city" not in st.session_state:
        st.session_state["ai_start_city"] = prof["home_city"]


def _logout():
    st.session_state.clear()


def render_auth_screen():
    st.markdown("""
    <style>
    section[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], [data-testid="collapsedControl"]{display:none!important}
    .auth-hero{min-height:540px;border-radius:22px;padding:38px 34px;color:#fff;display:flex;flex-direction:column;justify-content:flex-end;
        background:linear-gradient(180deg,rgba(20,35,12,.10),rgba(20,35,12,.90)),
        url('https://images.unsplash.com/photo-1506461883276-594a12b11cf3?auto=format&fit=crop&w=1400&q=80') center/cover}
    .auth-hero h1{color:#fff!important;font-size:clamp(2rem,4.5vw,3rem)!important;line-height:1.1;margin:0 0 10px!important}
    .auth-hero p{color:#F4F6EF!important;margin:0;max-width:430px;line-height:1.6}
    .auth-feats{margin-top:18px;display:flex;flex-wrap:wrap;gap:8px}
    .auth-feats span{background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.45);border-radius:20px;padding:5px 12px;font-size:13px;color:#fff}
    .st-key-auth_card{background:#fff;border:1px solid #DCE3D1;border-radius:20px;padding:26px 24px;box-shadow:0 10px 28px rgba(35,48,22,.10)}
    .st-key-auth_card .stRadio [role="radiogroup"]{gap:8px!important;min-height:0!important}
    .st-key-auth_card .stRadio [role="radiogroup"] label{background:#EEF1E7;border-radius:20px!important;padding:6px 16px!important}
    .st-key-auth_card .stRadio [role="radiogroup"] label p{font-size:1rem!important}
    @media(max-width:768px){.auth-hero{min-height:250px;padding:22px 20px}.st-key-auth_card{padding:20px 16px}}
    </style>
    """, unsafe_allow_html=True)
    left, right = st.columns([1.1, 1], gap="large")
    with left:
        st.markdown("""
        <div class="auth-hero">
          <h1>Your journey,<br>planned your way.</h1>
          <p>Log in to keep your favourites, saved trips, packing lists and special days in one place.</p>
          <div class="auth-feats"><span>🗓️ Smart itinerary</span><span>📅 Holiday calendar</span><span>🎒 Packing list</span><span>🚆 Transport</span></div>
        </div>
        """, unsafe_allow_html=True)
    with right:
        with st.container(key="auth_card"):
            mode = st.radio("Mode", ["Log in", "Sign up"], horizontal=True, key="auth_mode", label_visibility="collapsed")
            if mode == "Log in":
                st.markdown("### Welcome back 👋")
                with st.form("login_form"):
                    u = st.text_input("Username", key="login_user")
                    p = st.text_input("Password", type="password", key="login_pass")
                    go = st.form_submit_button("Log in", type="primary", use_container_width=True)
                if go:
                    if st.session_state.get("_auth_fails", 0) >= 5:
                        st.error("Too many wrong attempts. Refresh the page to try again.")
                    else:
                        prof = verify_user(u, p)
                        if prof:
                            _login_user(u.strip().lower(), prof)
                            st.rerun()
                        st.session_state["_auth_fails"] = st.session_state.get("_auth_fails", 0) + 1
                        st.error("Wrong username or password.")
            else:
                st.markdown("### Create your account ✨")
                with st.form("signup_form"):
                    n = st.text_input("Your name", key="su_name")
                    u = st.text_input("Choose a username", key="su_user", help="3-24 characters: letters, numbers, dot, underscore")
                    p1 = st.text_input("Password (6+ characters)", type="password", key="su_p1")
                    p2 = st.text_input("Confirm password", type="password", key="su_p2")
                    go = st.form_submit_button("Create account", type="primary", use_container_width=True)
                if go:
                    if p1 != p2:
                        st.error("The two passwords don't match.")
                    else:
                        legacy = load_favorites() if not USERS_PATH.exists() else None  # first account inherits old favourites
                        ok, msg = create_user(u, p1, n, legacy)
                        if ok:
                            _login_user(u.strip().lower(), verify_user(u, p1))
                            st.rerun()
                        st.error(msg)
            st.markdown("<div style='text-align:center;color:#758064;margin:10px 0 4px'>or</div>", unsafe_allow_html=True)
            if st.button("Continue as guest", key="guest_btn", use_container_width=True):
                _login_user("guest", {"name": "Guest", "avatar": "🧭", "home_city": "", "styles": [], "favorites": [],
                                      "saved_trips": [], "custom_days": [], "packing": {}}, guest=True)
                st.rerun()
            st.caption("Guest data disappears when you close the tab. Accounts are stored on this computer (demo login, not for production).")


if "user" not in st.session_state:
    render_auth_screen()
    st.stop()

# =====================================================
# TOP CONTROLS (sidebar-free layout)
# =====================================================
_u, _p = st.session_state["user"], st.session_state["profile"]
# Keep dataset administration available without occupying a permanent sidebar.
with st.popover("⚙️ Admin & data", use_container_width=False):
    uploaded = st.file_uploader("Upload tourist dataset (CSV)", type=["csv"], key="admin_dataset_upload")
    st.checkbox("Fetch missing place images from Wikipedia (may take longer)", value=True, key="wiki_images")
    st.caption(f"AI Smart Tourist Planner v{APP_VERSION}")

try:
    df = load_data(uploaded, st.session_state.get("extra_version", 0))
except Exception as error:
    st.error(f"Dataset error: {error}")
    st.stop()
df = apply_geo_fill(df)

all_states = ["All India"] + sorted(s for s in df["_state"].unique() if s and s != "Not specified")
if st.session_state.get("global_state", "All India") not in all_states:
    st.session_state["global_state"] = "All India"
state_col, admin_note_col = st.columns([2, 5])
with state_col:
    st.selectbox("📍 Focus app on a specific state", all_states, key="global_state")
selected_global_state = st.session_state.get("global_state", "All India")
with admin_note_col:
    st.caption(f"Explore {len(df):,} destinations across India · Change the state filter to personalize your results.")

if selected_global_state != "All India":
    active_df = df[df["_state"] == selected_global_state].copy()
else:
    active_df = df.copy()

# Show only category options that have matching places in the loaded CSV.
def categories_present_in_data(data):
    if data.empty or "_search_text" not in data.columns:
        return {}
    searchable = data["_search_text"].fillna("").astype(str).str.lower()
    available = {}
    for category, keywords in CATEGORIES.items():
        pattern = r"\b(?:" + "|".join(re.escape(word.lower()) for word in keywords) + r")\b"
        if searchable.str.contains(pattern, regex=True, na=False).any():
            available[category] = keywords
    return available


CATEGORIES = categories_present_in_data(active_df)
HOME_CHIPS = [
    ("🏖️ Beaches", "beach", "Beaches"),
    ("🏞️ Lakes & rivers", "lake", "Lakes & Rivers"),
    ("💧 Waterfalls", "waterfall", "Waterfalls"),
    ("🛕 Temples & religious", "temple", "Temples & Religious Places"),
    ("⛰️ Mountains & hills", "mountain", "Mountains & Hill Stations"),
    ("🏰 Forts & heritage", "fort", "Historical & Heritage"),
    ("🌳 Parks & nature", "park", "Nature, Parks & Gardens"),
    ("🏝️ Islands", "island", "Islands"),
    ("🏜️ Deserts", "desert", "Deserts"),
]
HOME_CHIPS = [chip for chip in HOME_CHIPS if chip[2] in CATEGORIES]

st.markdown("""
<style>
[data-testid="stSidebar"], [data-testid="stSidebarNav"], [data-testid="collapsedControl"] {display:none!important;}
[data-testid="stAppViewContainer"] > .main {margin-left:0!important;}
[data-testid="stHeader"] {background:transparent;}
.block-container {max-width:1440px;padding-top:1.2rem;padding-bottom:3rem;}
[data-testid="stPopover"] button {border-radius:14px;}
</style>
""", unsafe_allow_html=True)

# =====================================================
# FLOATING AI TRAVEL ASSISTANT (small round chat bubble)
# =====================================================
st.markdown("""
<style>
.st-key-chat_fab { position: fixed !important; right: 26px; bottom: 26px; z-index: 1000; width: auto !important; }
.st-key-chat_fab button {
    width: 62px !important; height: 62px !important; min-height: 62px !important; padding: 0 !important;
    border-radius: 50% !important; background: var(--accent) !important; border: none !important;
    box-shadow: 0 8px 22px rgba(0,0,0,.28) !important; transition: transform .15s ease;
}
.st-key-chat_fab button:hover { transform: scale(1.07); background: var(--accent-dark) !important; }
.st-key-chat_fab button p { font-size: 28px !important; margin: 0 !important; line-height: 1 !important; }
.st-key-chat_fab button [data-testid="stIconMaterial"] { display: none !important; }
div[data-testid="stPopoverBody"] { min-width: 370px; max-width: 94vw; }
</style>
""", unsafe_allow_html=True)


def render_chat_bubble():
    """Round chat button fixed at the bottom-right; opens a small chat window."""
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [{
            "role": "assistant",
            "content": "Hi! Ask me to plan a trip, suggest places, or look up fees, timings and best seasons.",
        }]
    with st.container(key="chat_fab"):
        with st.popover("💬", help="Ask the AI Travel Assistant"):
            st.markdown("**AI Travel Assistant**")
            st.caption("Answers come from your tourist dataset only.")
            top_state = active_df["_state"].value_counts().index[0] if len(active_df) else "India"
            quick = [f"Plan a 3 day trip in {top_state} under ₹15,000", f"Best time to visit {top_state}", f"Free or cheap places in {top_state}"]
            for q in quick:
                st.button(q, key=f"chip_{q}", use_container_width=True,
                          on_click=lambda q=q: st.session_state.update(chat_quick=q))
            chat_box = st.container(height=300)
            with st.form("chat_form", clear_on_submit=True):
                typed = st.text_input("Ask", placeholder="Ask about places, costs, timings...", label_visibility="collapsed")
                sent = st.form_submit_button("Send", use_container_width=True)
            prompt = st.session_state.pop("chat_quick", None) or (typed.strip() if sent and typed.strip() else None)
            if prompt:
                st.session_state["chat_messages"].append({"role": "user", "content": prompt})
                try:
                    reply, used = chat_answer(prompt, df, active_df)
                except Exception:
                    reply, used = "Sorry, I couldn't work that out from the dataset. Try rephrasing, for example: _plan a 3 day trip in Kerala_.", df.iloc[0:0]
                sources = used["_name"].head(8).tolist() if len(used) else []
                st.session_state["chat_messages"].append({"role": "assistant", "content": reply, "sources": sources})
            with chat_box:
                for msg in st.session_state["chat_messages"]:
                    with st.chat_message(msg["role"]):
                        st.markdown(msg["content"])
                        if msg.get("sources"):
                            st.caption("Dataset places used: " + ", ".join(msg["sources"]))
            if len(st.session_state["chat_messages"]) > 1:
                if st.button("🧹 Clear chat", key="chat_clear"):
                    del st.session_state["chat_messages"]
                    st.rerun()


# =====================================================
# 1. HOME
# =====================================================
def next_holiday(from_date, horizon=150):
    for i in range(horizon):
        d = from_date + datetime.timedelta(days=i)
        found = holidays_for_date(d)
        if found:
            return d, found[0][0]
    return None


def render_welcome():
    now = now_ist()
    prof, today = st.session_state["profile"], now.date()
    first = html.escape((prof.get("name") or "Traveller").split()[0])
    stubs = []
    nh = next_holiday(today)
    if nh:
        n = (nh[0] - today).days
        when = "today" if n == 0 else ("tomorrow" if n == 1 else f"in {n} days")
        stubs.append(("Next festival", f"{html.escape(nh[1])} · {when}"))
    s = st.session_state.get("itin_start_date")
    if st.session_state.get("trip_itinerary_summary") and isinstance(s, datetime.date) and s >= today:
        n = (s - today).days
        stubs.append(("Your trip", "starts today" if n == 0 else f"{n} day(s) to go"))
    stub_html = "".join(f'<div class="pass-stub"><div class="pass-k">{k}</div><div class="pass-v">{v}</div></div>' for k, v in stubs)
    clock = now.strftime("%I:%M %p").lstrip("0")
    st.markdown(
        f'<div class="pass"><div class="pass-main"><div class="pass-av">{html.escape(prof.get("avatar", "🧭"))}</div>'
        f'<div><div class="pass-hi">{greeting_for(now.hour)}, {first}!</div>'
        f'<div class="pass-sub">{now.strftime("%A, %d %B %Y")} · {clock} IST</div></div></div>{stub_html}</div>',
        unsafe_allow_html=True,
    )
    if not prof.get("home_city") and not st.session_state["user"]["guest"]:
        st.caption("Tip: add your home city under My Travel → Profile and it becomes your starting point everywhere.")


def open_home_feature(page_key, section_key=None, section_value=None):
    """Open a homepage feature and select its matching subsection."""
    if section_key and section_value:
        st.session_state[section_key] = section_value
    st.switch_page(PAGES[page_key])


def page_home():
    render_welcome()
    render_hero_banner(selected_global_state, len(active_df))

    st.markdown("### ✨ Explore Our Smart Travel Features")
    st.caption("Everything you need to discover destinations and plan a memorable trip.")
    tiles = [
        ("ai", None, None, "🤖", "AI Recommendations", "Personalized places based on your interests and budget."),
        ("explore", "sec_explore", EXPLORE_SECTIONS[0], "📍", "Destination Explorer", "Search and filter destinations across India."),
        ("explore", "sec_explore", EXPLORE_SECTIONS[1], "🧭", "Place Details & Transport", "Images, timings, maps and nearby transport connections."),
        ("plan", "sec_plan", PLAN_SECTIONS[0], "🗓️", "Itinerary Planner", "Build and edit your day-wise travel plan."),
        ("plan", "sec_plan", PLAN_SECTIONS[2], "💰", "Budget Calculator", "Estimate trip costs and track group expenses."),
        ("plan", "sec_plan", PLAN_SECTIONS[1], "🎒", "Smart Packing List", "Prepare a trip-ready packing checklist."),
        ("mytrip", "sec_mytrip", MYTRIP_SECTIONS[0], "⭐", "My Travel", "Favourites, saved trips, profile and summary."),
    ]
    for start in range(0, len(tiles), 4):
        cols = st.columns(4)
        for col, tile in zip(cols, tiles[start:start + 4]):
            key, section_key, section_value, icon, title, description = tile
            with col:
                st.markdown(
                    f'<div class="home-feature"><div class="feature-icon">{icon}</div><h4>{title}</h4><p>{description}</p></div>',
                    unsafe_allow_html=True,
                )
                st.button("Explore feature  →", key=f"home_feature_{start}_{key}_{title.replace(' ', '_')}",
                          use_container_width=True, on_click=open_home_feature,
                          args=(key, section_key, section_value))

    st.markdown("### 🧳 Discover destinations")
    st.caption("Popular places selected from your destination dataset.")
    trending = chat_rank(active_df).head(4)
    if trending.empty:
        st.info("No destinations available for this selection.")
    else:
        cols = st.columns(4)
        for idx, (_, place) in enumerate(trending.iterrows()):
            with cols[idx]:
                render_place_card(place, "home_trend", idx)
        st.page_link(PAGES["explore"], label="Explore all destinations", icon="📍")

    st.markdown(f"<p style='text-align:center;color:#758064;font-size:12px;margin-top:30px;'>AI Smart Tourist Planner v{APP_VERSION} · {len(active_df)} destinations</p>", unsafe_allow_html=True)
    render_chat_bubble()

# =====================================================
# 2. AI RECOMMENDATIONS
# =====================================================

from math import radians, sin, cos, sqrt, atan2

def get_city_coordinates(city):
    """(lat, lon) for a city: dataset places first, then an online lookup (cached)."""
    name = str(city or "").strip()
    if not name:
        return None
    known = df[df["_city"].str.lower() == name.lower()].dropna(subset=["_latitude", "_longitude"])
    if len(known):
        return float(known["_latitude"].median()), float(known["_longitude"].median())
    return geocode_cached(name, "", "")


def calculate_distance_km(start_city, destination_city, destination_state=""):

    """
    Calculate approximate distance between two cities
    using their latitude and longitude.
    """

    try:
        # Find starting city coordinates
        start = get_city_coordinates(start_city)

        # Find destination coordinates
        destination = get_city_coordinates(destination_city)

        if not start or not destination:
            return None

        lat1, lon1 = start
        lat2, lon2 = destination

        # Earth's radius in kilometres
        R = 6371.0

        lat1 = radians(float(lat1))
        lon1 = radians(float(lon1))
        lat2 = radians(float(lat2))
        lon2 = radians(float(lon2))

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            sin(dlat / 2) ** 2
            + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        )

        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        return round(R * c, 2)

    except Exception:
        return None

def page_ai():

    st.header(f"🤖 AI Trip Recommendations — {selected_global_state}")

    st.write(
        "Find destinations based on your interests, budget, "
        "travel distance, season and traveller preferences."
    )

    # -------------------------------------------------
    # USER PREFERENCES
    # -------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        keep("ai_start_city", "Kolkata")

        start_city = st.text_input(
            "Your starting city",
            key="ai_start_city"
        )

        keep(
            "ai_month",
            MONTHS[datetime.date.today().month - 1],
            MONTHS
        )

        month_name = st.selectbox(
            "Travel month",
            MONTHS,
            key="ai_month"
        )

        traveler = st.selectbox(
            "Who is travelling?",
            list(TRAVELER_PROFILES.keys()),
            key="ai_traveler"
        )

        keep("ai_days", 3)

        trip_days = st.number_input(
            "Trip duration (days)",
            min_value=1,
            max_value=14,
            key="ai_days"
        )

    with col2:

        tier = st.selectbox(
            "Budget style",
            list(BUDGET_PRESETS.keys()),
            key="ai_tier"
        )

        reach_label = st.selectbox(
            "How far will you travel?",
            list(REACH_KM.keys()),
            key="ai_reach"
        )

        keep(
            "ai_multi_category",
            ["Historical & Heritage"],
            list(CATEGORIES.keys()),
            multi=True
        )

        selected_categories = st.multiselect(
            "Categories you enjoy",
            list(CATEGORIES.keys()),
            key="ai_multi_category"
        )

        query = st.text_input(
            "Describe your ideal trip (optional)",
            placeholder="peaceful, trekking, old architecture",
            key="ai_query"
        )

    keep("ai_top_n", 5)

    top_n = st.slider(
        "Number of recommendations",
        min_value=3,
        max_value=15,
        key="ai_top_n"
    )

    # -------------------------------------------------
    # GENERATE RECOMMENDATIONS
    # -------------------------------------------------

    if st.button(
        "✨ Find My Best Matches",
        type="primary",
        key="ai_recommend_button"
    ):

        pool = active_df.reset_index(drop=True).copy()

        if pool.empty:

            st.warning(
                f"No destinations available in {selected_global_state}."
            )

            return

        profile = TRAVELER_PROFILES[traveler]

        # -------------------------------------------------
        # CATEGORY FILTERING
        # -------------------------------------------------

        if selected_categories:

            category_keywords = []

            for category in selected_categories:
                category_keywords.extend(
                    CATEGORIES.get(category, [])
                )

            category_pattern = "|".join(
                re.escape(str(word).lower())
                for word in category_keywords
            )

            category_mask = pool["_search_text"].fillna("").str.lower().str.contains(
                category_pattern,
                regex=True,
                na=False
            )

            category_pool = pool[category_mask].copy()

            if not category_pool.empty:
                pool = category_pool

        # -------------------------------------------------
        # DISTANCE FILTERING
        # -------------------------------------------------

        max_distance = REACH_KM.get(reach_label)

        if max_distance is not None and start_city.strip():

            if "_distance_km" not in pool.columns:

                pool["_distance_km"] = np.nan

            # Use the existing distance function if available.
            # Otherwise, keep destinations without known distance
            # and clearly label their distance as unavailable.

            if "calculate_distance_km" in globals():

                pool["_distance_km"] = pool.apply(
                    lambda r: calculate_distance_km(
                        start_city,
                        r.get("_city", ""),
                        r.get("_state", "")
                    ),
                    axis=1
                )

                pool = pool[
                    pool["_distance_km"].isna()
                    | (pool["_distance_km"] <= max_distance)
                ].copy()

        # -------------------------------------------------
        # PERSONALIZED SCORING
        # -------------------------------------------------

        results = []

        month_index = MONTHS.index(month_name)

        for _, row in pool.iterrows():

            score = 0.0
            reasons = []

            search_text = str(
                row.get("_search_text", "")
            ).lower()

            # Category score
            if selected_categories:

                matched = []

                for category in selected_categories:

                    keywords = CATEGORIES.get(category, [])

                    if any(
                        str(word).lower() in search_text
                        for word in keywords
                    ):
                        matched.append(category)

                if matched:
                    score += 30
                    reasons.append(
                        "Matches your interests: "
                        + ", ".join(matched)
                    )

            else:
                score += 15
                reasons.append("General destination recommendation")

            # Rating score
            rating = row.get("_rating")

            if pd.notna(rating):

                rating_score = max(
                    0,
                    min(float(rating) / 5 * 20, 20)
                )

                score += rating_score

                if float(rating) >= 4.0:
                    reasons.append("Highly rated by visitors")

            # Popularity score
            reviews = row.get("_reviews")

            if pd.notna(reviews):

                review_score = min(
                    np.log1p(max(float(reviews), 0)) / 10 * 15,
                    15
                )

                score += review_score

                if float(reviews) > 0:
                    reasons.append("Has visitor reviews")

         # Budget score
            fee = row.get("_fee")

            if pd.notna(fee):

                fee_value = float(fee)

                if fee_value <= 0:
                    score += 15
                    reasons.append("Free entry")
                else:
                    score += 3

            # Season score
            best_time = str(
                row.get("_best_time", "")
            ).strip()

            if best_time and best_time.lower() not in ("nan", "all"):

                if season_fit_place(
                    best_time,
                    row.get("_type", ""),
                    month_index
                ) == 1.0:

                    score += 10
                    reasons.append("Good season to visit")

            else:
                score += 5

            # Description/query matching
            if query.strip():

                query_words = [
                    word.lower()
                    for word in re.findall(r"\w+", query)
                    if len(word) > 2
                ]

                matches = [
                    word for word in query_words
                    if word in search_text
                ]

                if matches:

                    score += min(
                        len(matches) * 3,
                        15
                    )

                    reasons.append(
                        "Matches your description: "
                        + ", ".join(matches[:3])
                    )

            # Traveller profile
            profile_text = str(profile).lower()

            if any(
                word in search_text
                for word in profile_text.split()
                if len(word) > 3
            ):
                score += 5
                reasons.append("Relevant to your traveller profile")

            # Distance bonus
            distance = row.get("_distance_km")

            if pd.notna(distance):

                if distance <= 100:
                    score += 5
                    reasons.append("Convenient nearby destination")

            score = min(score, 100)

            results.append({
                "row": row,
                "score": score,
                "reasons": reasons
            })

        # -------------------------------------------------
        # SORT RESULTS
        # -------------------------------------------------

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        results = results[:int(top_n)]

        if not results:

            st.warning(
                "No matching destinations found. "
                "Try changing your interests or travel distance."
            )

            return

        # -------------------------------------------------
        # RESULTS
        # -------------------------------------------------

        st.success(
            f"Found {len(results)} personalized recommendations!"
        )

        st.markdown("### 🌟 Your Recommended Destinations")

        for index, item in enumerate(results, start=1):

            row = item["row"]
            score = item["score"]
            reasons = item["reasons"]

            st.markdown(
                f"### {index}. {row['_name']}"
            )

            st.progress(
                int(score),
                text=f"AI Match Score: {score:.1f}%"
            )

            # Destination card
            render_place_card(
                row,
                "ai_recommendation",
                index,
                image_width=190
            )

            # Recommendation explanation
            st.markdown(
                "**💡 Why the AI recommended this destination**"
            )

            for reason in reasons:
                st.markdown(f"- {reason}")

            distance = row.get("_distance_km")

            if pd.notna(distance):
                st.caption(
                    f"Approximate distance from {start_city}: "
                    f"{distance:.1f} km"
                )
            else:
                st.caption("Distance information is unavailable.")

            st.divider()
# =====================================================
# 3. DESTINATION EXPLORER
# =====================================================
def page_explore():
    st.header(f"📍 Destination Explorer — {selected_global_state}")
    sk = selected_global_state
    fee_series = active_df["_fee"].dropna()
    fee_cap = int(max(100, math.ceil((fee_series.max() if len(fee_series) else 100) / 50) * 50))

    e1, e2 = st.columns(2)
    with e1:
        search_query = st.text_input("Search places by name, city, or feature", placeholder="Taj Mahal, beach, temple...", key="ex_search")
    with e2:
        type_options = sorted(t for t in active_df["_type"].unique() if t)
        keep(f"ex_types_{sk}", [], type_options, multi=True)
        chosen_types = st.multiselect("Dataset place type", type_options, key=f"ex_types_{sk}")

    category_options = list(CATEGORIES.keys())
    keep(f"ex_categories_{sk}", [], category_options, multi=True)
    chosen_categories = st.multiselect(
        "Explore by category",
        category_options,
        key=f"ex_categories_{sk}",
        placeholder="Choose beaches, lakes, temples, mountains..."
    )

    e3, e4, e5 = st.columns(3)
    with e3:
        keep("ex_rating", 0.0)
        min_rating = st.slider("Minimum rating", 0.0, 5.0, step=0.5, key="ex_rating")
    with e4:
        keep_range(f"ex_fee_{sk}", fee_cap, 0, fee_cap)
        max_fee = st.slider("Maximum entry fee (₹)", 0, fee_cap, step=50, key=f"ex_fee_{sk}")
    with e5:
        sort_by = st.selectbox("Sort by", ["Best rated", "Most reviewed", "Name A–Z", "Lowest fee"], key="ex_sort")
    keep("ex_unlisted", True)
    keep_unlisted = st.checkbox("Keep places with no rating or fee listed", key="ex_unlisted")

    filtered_df = active_df
    if search_query:
        sq = search_query.lower()
        filtered_df = filtered_df[
            filtered_df["_name"].str.lower().str.contains(sq, na=False, regex=False) |
            filtered_df["_city"].str.lower().str.contains(sq, na=False, regex=False) |
            filtered_df["_type"].str.lower().str.contains(sq, na=False, regex=False) |
            filtered_df["_description"].str.lower().str.contains(sq, na=False, regex=False)
        ]
    if chosen_types:
        filtered_df = filtered_df[filtered_df["_type"].isin(chosen_types)]
    if chosen_categories:
        category_words = sorted({
            word
            for category in chosen_categories
            for word in CATEGORIES.get(category, [])
        })
        category_pattern = r"\b(?:" + "|".join(re.escape(word) for word in category_words) + r")\b"
        category_match = filtered_df["_search_text"].fillna("").str.contains(
            category_pattern, case=False, regex=True, na=False
        )
        filtered_df = filtered_df[category_match]
    rating_ok = filtered_df["_rating"] >= min_rating
    fee_ok = filtered_df["_fee"] <= max_fee
    if keep_unlisted:
        rating_ok = rating_ok | filtered_df["_rating"].isna()
        fee_ok = fee_ok | filtered_df["_fee"].isna()
    filtered_df = filtered_df[rating_ok & fee_ok]
    sort_map = {
        "Best rated": ("_rating", False), "Most reviewed": ("_reviews", False),
        "Name A–Z": ("_name", True), "Lowest fee": ("_fee", True),
    }
    col_, asc_ = sort_map[sort_by]
    filtered_df = filtered_df.sort_values(col_, ascending=asc_, na_position="last")
    st.caption(f"{len(filtered_df)} of {len(active_df)} places match your filters.")

    ex_map, ex_cards, ex_near = st.tabs(["🗺️ Map view", "🗂️ Place cards", "📍 Nearby places"])

    with ex_map:
        if filtered_df.empty:
            st.info("No places match these filters.")
        else:
            mapped, missing = render_explorer_map(filtered_df, key="explorer_main_map")
            _ap = st.session_state.get("_map_approx", 0)
            st.caption(
                f"{mapped} places on the map"
                + (f" ({_ap} approximate, placed at the city centre)" if _ap else "")
                + (f"; {missing} cannot be mapped yet because no place in their city has coordinates" if missing else "")
                + ". Pin colour follows place type, grey pins are approximate, click a cluster to zoom in."
            )
            _need = filtered_df[filtered_df["_latitude"].isna() | filtered_df["_longitude"].isna()]
            if len(_need):
                if st.button(f"📍 Find exact locations for {min(len(_need), 40)} of {len(_need)} place(s)", key="ex_geo_fill",
                             help="Looks places up online (about a second each) and remembers them, so this is a one-time job. Press again for the next batch."):
                    _found = fill_exact_locations(_need.head(40))
                    st.session_state["_geo_msg"] = f"Found exact locations for {_found} place(s). Press the button again for the next batch."
                    st.rerun()
            if st.session_state.get("_geo_msg"):
                st.success(st.session_state.pop("_geo_msg"))
    with ex_cards:
        if filtered_df.empty:
            st.info("No destinations match your filters. Please change your search or filters.")
        else:
            max_cards = min(60, len(filtered_df))
            default_cards = min(12, max_cards)

            if max_cards <= 1:
                # Streamlit sliders require min_value < max_value.
                # With one matching destination, show that single card directly.
                show_n = 1
                st.caption("Showing the only matching destination.")
                st.session_state["ex_cards_n"] = 1
            else:
                keep_range("ex_cards_n", default_cards, 1, max_cards)
                show_n = st.slider(
                    "Cards to show",
                    min_value=1,
                    max_value=max_cards,
                    key="ex_cards_n",
                )

            layout = st.radio("Card layout", ["Grid", "Detailed list"], horizontal=True, key="ex_layout")
            card_rows = list(filtered_df.head(show_n).iterrows())
            if layout == "Grid":
                for start in range(0, len(card_rows), 3):
                    grid_cols = st.columns(3)
                    for g_col, (n_, (_, r_)) in zip(grid_cols, enumerate(card_rows[start:start + 3], start=start)):
                        with g_col:
                            render_place_card(r_, "explorer", n_)
            else:
                for _, row in card_rows:
                    show_destination(row, tab_name="explorer")

    with ex_near:
        st.write("Pick a place and see what else is within reach. Search covers the whole dataset, not just the selected state.")
        labels = {}
        for i, r in active_df.iterrows():
            labels[f"{r['_name']} — {r['_city']}" if r["_city"] else r["_name"]] = i
        if not labels:
            st.info("No places available.")
        else:
            n1, n2, n3 = st.columns([3, 2, 2])
            with n1:
                keep("near_pick", list(labels)[0], list(labels))
                pick = st.selectbox("Place you are visiting", list(labels), key="near_pick")
            with n2:
                keep("near_radius", 50)
                radius = st.slider("Radius (km)", 5, 300, step=5, key="near_radius")
            with n3:
                near_types = st.multiselect("Only these types", sorted(t for t in df["_type"].unique() if t), key="near_types")
            base_row = df.loc[labels[pick]].copy()
            if pd.isna(base_row["_latitude"]) or pd.isna(base_row["_longitude"]):
                with st.spinner("This place has no coordinates in the dataset; looking it up..."):
                    found = geocode_cached(base_row["_name"], base_row["_city"], base_row["_state"])
                if found:
                    base_row["_latitude"], base_row["_longitude"] = found
                else:
                    st.warning("Couldn't find coordinates for this place, so nearby search is unavailable for it.")
            near = nearby_places(df, base_row, radius, near_types or None, limit=60)
            if pd.notna(base_row["_latitude"]):
                if near.empty:
                    st.info(f"No other places in the dataset within {radius} km. Try a larger radius.")
                else:
                    render_explorer_map(near, center_row=base_row, radius_km=radius, key="explorer_near_map")
                    table = pd.DataFrame({
                        "Place": near["_name"], "City": near["_city"], "Type": near["_type"],
                        "Distance (km)": near["_dist_km"].round(1),
                        "≈ Drive (h)": (near["_dist_km"] * 1.3 / 40).round(1),
                        "Entry fee (₹)": near["_fee"], "Rating": near["_rating"],
                    })
                    st.dataframe(table, hide_index=True, use_container_width=True)
                    st.caption("Distance is straight-line; drive time assumes road distance ≈ 1.3× at 40 km/h.")
                    with st.expander("Show these as cards"):
                        for _, row in near.head(10).iterrows():
                            show_destination(row, tab_name="nearby")


# =====================================================
# 5. TRAVEL BUDGET (group planner with shared expenses)
# =====================================================
TRIP_COMPANIONS = {
    "Solo": ["Me"],
    "Couple": ["Partner 1", "Partner 2"],
    "Family": ["Member 1", "Member 2", "Member 3", "Member 4"],
    "Friends": ["Friend 1", "Friend 2", "Friend 3", "Friend 4"],
    "Colleagues": ["Colleague 1", "Colleague 2", "Colleague 3", "Colleague 4", "Colleague 5"],
}
SHARE_METHODS = ["Equal sharing", "Custom percentage sharing"]
EXPENSE_CATEGORIES = [
    "Hotel / Stay", "Food & Dining", "Local Commute",
    "Intercity Transport", "Entry Fees", "Shopping", "Other",
]
AMT = "Amount (₹)"


def split_paise(total_paise, weights):
    """Split an integer amount (in paise) by weights so the parts add up exactly."""
    total_weight = sum(weights)
    raw = [total_paise * w / total_weight for w in weights]
    parts = [int(math.floor(r)) for r in raw]
    leftover = total_paise - sum(parts)
    by_fraction = sorted(range(len(raw)), key=lambda i: raw[i] - parts[i], reverse=True)
    for i in by_fraction[:leftover]:
        parts[i] += 1
    return parts


def settle_up(balances):
    """Fewest-ish payments that clear every balance. balances: {name: paise}, + = is owed."""
    creditors = sorted([[n, b] for n, b in balances.items() if b > 0], key=lambda x: -x[1])
    debtors = sorted([[n, -b] for n, b in balances.items() if b < 0], key=lambda x: -x[1])
    transfers, i, j = [], 0, 0
    while i < len(debtors) and j < len(creditors):
        pay = min(debtors[i][1], creditors[j][1])
        transfers.append((debtors[i][0], creditors[j][0], pay))
        debtors[i][1] -= pay
        creditors[j][1] -= pay
        if debtors[i][1] == 0:
            i += 1
        if creditors[j][1] == 0:
            j += 1
    return transfers


def build_estimate_df(days, travelers, tier, dist_km, mode, entry_pp=None):
    rates = BUDGET_PRESETS[tier]
    avg_fee = active_df["_fee"].dropna().mean()
    entry = entry_pp * travelers if entry_pp is not None else (avg_fee if pd.notna(avg_fee) else 100) * 3 * travelers
    rows = [
        ("Hotel / Stay", f"Estimated stay ({math.ceil(travelers / 2)} room(s))", rates["hotel"] * days * math.ceil(travelers / 2)),
        ("Food & Dining", "Estimated food", rates["food"] * days * travelers),
        ("Local Commute", "Estimated local travel", rates["local"] * days * travelers),
        ("Intercity Transport", f"Estimated {mode.split(' (')[0].lower()} travel",
         dist_km * TRANSPORT_RATES[mode] * (1 if mode in PER_VEHICLE_MODES else travelers)),
        ("Entry Fees", "Estimated entry fees", entry),
    ]
    return pd.DataFrame({
        "Category": [r[0] for r in rows],
        "Description": [r[1] for r in rows],
        AMT: [round(float(r[2]), 2) for r in rows],
        "Paid by": [None] * len(rows),
    })


def money(paise):
    return f"₹{paise / 100:,.2f}"


def render_trip_budget():
    arrived = st.session_state.get("_arrived", False)
    st.session_state.setdefault("tb_days", 4)
    st.session_state.setdefault("tb_dist", 500)
    st.header("💰 Smart Travel Budget Calculator")
    st.caption(
        "Equal sharing decides everyone's fair share. Who actually paid decides "
        "who owes money and who gets money back."
    )

    # ---- Link from the itinerary planner -----------------------------
    ib = st.session_state.get("itin_budget")
    if ib:
        with st.container(border=True):
            using = st.session_state.get("tb_entry_pp") is not None
            st.markdown(
                f"🔗 **Itinerary linked:** {ib['days']} day(s) in {ib['city']}, {len(ib['stops'])} stops, "
                f"entry fees ₹{ib['entry_pp']:,.0f} per person"
                + (f" ({ib['unlisted']} stop(s) with no listed fee)." if ib["unlisted"] else ".")
                + ("  \n✅ Currently used in the estimate below." if using else "")
            )
            bl1, bl2 = st.columns(2)
            if bl1.button("Use itinerary data in this budget", key="tb_use_itin"):
                st.session_state["tb_days"] = int(ib["days"])
                st.session_state["tb_entry_pp"] = float(ib["entry_pp"])
                st.session_state["tb_exp_sig"] = None
            if using and bl2.button("Stop using itinerary data", key="tb_stop_itin"):
                st.session_state.pop("tb_entry_pp", None)
                st.session_state["tb_exp_sig"] = None

    tr = st.session_state.get("trip_route")
    if tr:
        _factor = {"Bus": 1.3, "Train": 1.2, "Flight": 1.0}.get(tr["best"], 1.3)
        _rt_km = int(round(tr["straight_km"] * _factor * 2))
        _mode_key = {"Bus": "Bus (road-distance estimate)", "Train": "Train (approximate)", "Flight": "Flight (approximate)"}[tr["best"]]
        with st.container(border=True):
            st.markdown(
                f"🔗 **Route from the Transport section:** {tr['start']} → {tr['dest']}, about {_rt_km:,} km round trip. "
                f"Suggested mode: **{tr['best']}**."
            )
            if _rt_km > 3000:
                st.caption("Distance is capped at 3,000 km in this calculator.")
            if st.button("Use route distance and suggested mode", key="tb_use_route"):
                st.session_state["tb_dist"] = min(max(_rt_km, 50), 3000)
                st.session_state["tb_mode"] = _mode_key
                st.session_state["tb_exp_sig"] = None

    # ---- Step 1: who + trip basics -----------------------------------
    st.subheader("1️⃣ Who are you travelling with?")
    group = st.radio(
        "Travelling with", list(TRIP_COMPANIONS.keys()), horizontal=True,
        key="tb_group", label_visibility="collapsed",
    )
    default_names = TRIP_COMPANIONS[group]

    c1, c2 = st.columns(2)
    with c1:
        b_days = st.number_input("Number of Days", 1, 30, key="tb_days")
        b_tier = st.selectbox("Budget Tier", list(BUDGET_PRESETS.keys()), key="tb_tier")
    with c2:
        b_dist_km = st.number_input("Approximate Round-Trip Distance (km)", 50, 3000, key="tb_dist")
        b_mode = st.selectbox("Primary Transport Mode", list(TRANSPORT_RATES.keys()), key="tb_mode")
        _one_way = b_dist_km / 2
        _hint = ("Bus or car usually works best" if _one_way < 300
                 else "Train is usually the best value" if _one_way <= 1000
                 else "Flight or an overnight/long-distance train is usually best")
        st.caption(f"💡 For about {_one_way:,.0f} km each way: {_hint}. Car cost is per vehicle; other modes are per person.")

    # ---- Step 2: sharing method --------------------------------------
    st.subheader("2️⃣ How will you share the budget?")
    if group == "Solo":
        method = SHARE_METHODS[0]
        st.info("Solo trip: you cover everything, so there is nothing to split.")
    else:
        method = st.radio("Sharing method", SHARE_METHODS, horizontal=True, key="tb_method")
        if method == SHARE_METHODS[0]:
            st.caption("Everyone's fair share is the same. You'll still record who paid each expense.")
        else:
            st.caption("Set each person's share in %, e.g. 60/40 if one person uses more of the trip. Shares must total 100%.")

    # ---- Step 3: members and expenses --------------------------------
    st.subheader("3️⃣ Members and expenses")
    custom = method == SHARE_METHODS[1]
    members_default = pd.DataFrame({
        "Name": default_names,
        "Share %": [round(100 / len(default_names), 2)] * len(default_names),
    })
    st.markdown("**Members**")
    _mb_key = f"tb_mbase_{group}"
    if arrived or _mb_key not in st.session_state:
        st.session_state[_mb_key] = st.session_state.get(f"tb_mlive_{group}", members_default)
    members_df = st.data_editor(
        st.session_state[_mb_key],
        num_rows="fixed" if group == "Solo" else "dynamic",
        key=f"tb_members_{group}",
        hide_index=True,
        use_container_width=True,
        column_order=["Name", "Share %"] if custom else ["Name"],
        column_config={
            "Name": st.column_config.TextColumn("Member name", required=True),
            "Share %": st.column_config.NumberColumn("Share %", min_value=0.0, max_value=100.0, step=0.5, format="%.2f"),
        },
    )

    st.session_state[f"tb_mlive_{group}"] = members_df.copy()
    members = members_df.dropna(subset=["Name"]).copy()
    members["Name"] = members["Name"].astype(str).str.strip()
    members = members[members["Name"] != ""]
    names = members["Name"].tolist()
    if not names:
        st.warning("Add at least one member to continue.")
        return
    if len(set(names)) != len(names):
        st.error("Member names must be unique so payments can be tracked correctly.")
        return

    # Expense table: estimated rows pre-filled, user edits to real amounts
    if arrived and st.session_state.get("tb_exp_live") is not None:
        st.session_state["tb_exp_init"] = st.session_state["tb_exp_live"]
        st.session_state["tb_exp_sig"] = st.session_state.get("tb_exp_live_group", group)
        st.session_state["tb_exp_ver"] = st.session_state.get("tb_exp_ver", 0) + 1
    if st.session_state.get("tb_exp_sig") != group or "tb_exp_init" not in st.session_state:
        st.session_state["tb_exp_init"] = build_estimate_df(
            b_days, len(default_names), b_tier, b_dist_km, b_mode, st.session_state.get("tb_entry_pp"))
        st.session_state["tb_exp_sig"] = group
        st.session_state["tb_exp_ver"] = st.session_state.get("tb_exp_ver", 0) + 1

    head_l, head_r = st.columns([3, 1])
    head_l.markdown("**Expenses**  ·  rows start as estimates: edit amounts, add rows, and pick who paid")
    if head_r.button("🔄 Refill from trip details", key="tb_refill", use_container_width=True):
        st.session_state["tb_exp_init"] = build_estimate_df(
            b_days, len(names), b_tier, b_dist_km, b_mode, st.session_state.get("tb_entry_pp"))
        st.session_state["tb_exp_ver"] += 1

    expenses_df = st.data_editor(
        st.session_state["tb_exp_init"],
        num_rows="dynamic",
        key=f"tb_exp_{st.session_state['tb_exp_ver']}",
        hide_index=True,
        use_container_width=True,
        column_config={
            "Category": st.column_config.SelectboxColumn("Category", options=EXPENSE_CATEGORIES, required=True),
            "Description": st.column_config.TextColumn("Description"),
            AMT: st.column_config.NumberColumn(AMT, min_value=0.0, step=50.0, format="%.2f"),
            "Paid by": st.column_config.SelectboxColumn("Paid by", options=names),
        },
    )

    st.session_state["tb_exp_live"] = expenses_df.copy()
    st.session_state["tb_exp_live_group"] = group
    exp = expenses_df.copy()
    exp[AMT] = pd.to_numeric(exp[AMT], errors="coerce")
    exp = exp[exp[AMT].notna() & (exp[AMT] > 0)].copy()
    exp["Category"] = exp["Category"].fillna("Other")
    if group == "Solo":
        exp["Paid by"] = names[0]
    if exp.empty:
        st.info("Enter at least one expense amount to see the results.")
        return
    missing_payer = ~exp["Paid by"].isin(names)
    if missing_payer.any():
        st.warning(
            f"{int(missing_payer.sum())} expense(s) have no valid 'Paid by' member. "
            "Choose who paid each one (renamed a member? re-select them)."
        )
        return

    # Sharing weights
    if custom:
        pct = pd.to_numeric(members["Share %"], errors="coerce")
        if pct.isna().any() or abs(pct.sum() - 100) > 0.01:
            st.error(f"Custom shares must total 100% (currently {pct.fillna(0).sum():.2f}%).")
            return
        weights = pct.tolist()
    else:
        weights = [1] * len(names)

    # ---- Step 4: contributions and balances --------------------------
    st.divider()
    st.subheader("4️⃣ Contributions and balances")
    exp["_paise"] = (exp[AMT] * 100).round().astype(int)
    paid = {n: int(exp.loc[exp["Paid by"] == n, "_paise"].sum()) for n in names}
    total_paise = sum(paid.values())
    shares = split_paise(total_paise, weights)
    balances = {n: paid[n] - s for n, s in zip(names, shares)}

    def status(b):
        if b > 0:
            return f"Gets back {money(b)}"
        if b < 0:
            return f"Owes {money(-b)}"
        return "Settled"

    result = pd.DataFrame({
        "Member": names,
        "Share %": [round(100 * w / sum(weights), 2) for w in weights],
        "Fair share (₹)": [s / 100 for s in shares],
        "Actually paid (₹)": [paid[n] / 100 for n in names],
        "Balance (₹)": [balances[n] / 100 for n in names],
        "Status": [status(balances[n]) for n in names],
    })
    st.dataframe(
        result, hide_index=True, use_container_width=True,
        column_config={
            "Share %": st.column_config.NumberColumn(format="%.2f"),
            "Fair share (₹)": st.column_config.NumberColumn(format="%.2f"),
            "Actually paid (₹)": st.column_config.NumberColumn(format="%.2f"),
            "Balance (₹)": st.column_config.NumberColumn(format="%+.2f"),
        },
    )
    st.caption("Balance = actually paid − fair share. Positive means the group owes them; negative means they owe the group.")

    # ---- Step 5: pie chart and final budget --------------------------
    st.divider()
    st.subheader("5️⃣ Final trip budget")
    m1, m2, m3 = st.columns(3)
    with m1: render_metric_card("Total Trip Cost", money(total_paise), "💳")
    with m2: render_metric_card("Average Per Person", money(total_paise / len(names)), "👤")
    with m3: render_metric_card("Daily Average", money(total_paise / b_days), "📅")

    by_cat = (exp.groupby("Category")["_paise"].sum().reset_index()
              .rename(columns={"_paise": "paise"}).sort_values("paise", ascending=False))
    by_cat[AMT] = by_cat["paise"] / 100
    by_cat["Share of total"] = (by_cat["paise"] / total_paise * 100).round(1)

    pie_col, table_col = st.columns([3, 2])
    with pie_col:
        st.vega_lite_chart(
            by_cat[["Category", AMT, "Share of total"]],
            {
                "mark": {"type": "arc", "innerRadius": 55, "outerRadius": 130},
                "encoding": {
                    "theta": {"field": AMT, "type": "quantitative"},
                    "color": {"field": "Category", "type": "nominal", "legend": {"orient": "bottom"}},
                    "tooltip": [
                        {"field": "Category", "type": "nominal"},
                        {"field": AMT, "type": "quantitative", "format": ",.2f"},
                        {"field": "Share of total", "type": "quantitative", "title": "% of total"},
                    ],
                },
            },
            use_container_width=True,
        )
    with table_col:
        st.dataframe(
            by_cat[["Category", AMT, "Share of total"]].rename(columns={"Share of total": "% of total"}),
            hide_index=True, use_container_width=True,
            column_config={
                AMT: st.column_config.NumberColumn(format="%.2f"),
                "% of total": st.column_config.NumberColumn(format="%.1f%%"),
            },
        )

    st.markdown("**🧾 Where the money was used (every expense)**")
    items = exp[["Category", "Description", AMT, "Paid by"]].sort_values(AMT, ascending=False)
    st.dataframe(
        items, hide_index=True, use_container_width=True,
        column_config={AMT: st.column_config.NumberColumn(format="%.2f")},
    )

    st.markdown("### 🤝 Who paid how much")
    contrib = result[["Member", "Actually paid (₹)", "Fair share (₹)", "Balance (₹)", "Status"]]
    who_l, who_r = st.columns([2, 3])
    with who_l:
        paid_only = contrib[contrib["Actually paid (₹)"] > 0][["Member", "Actually paid (₹)"]]
        st.vega_lite_chart(
            paid_only,
            {
                "mark": {"type": "arc", "innerRadius": 45, "outerRadius": 110},
                "encoding": {
                    "theta": {"field": "Actually paid (₹)", "type": "quantitative"},
                    "color": {"field": "Member", "type": "nominal", "legend": {"orient": "bottom"}},
                    "tooltip": [
                        {"field": "Member", "type": "nominal"},
                        {"field": "Actually paid (₹)", "type": "quantitative", "format": ",.2f"},
                    ],
                },
            },
            use_container_width=True,
        )
    with who_r:
        st.dataframe(
            contrib, hide_index=True, use_container_width=True,
            column_config={
                "Actually paid (₹)": st.column_config.NumberColumn(format="%.2f"),
                "Fair share (₹)": st.column_config.NumberColumn(format="%.2f"),
                "Balance (₹)": st.column_config.NumberColumn(format="%+.2f"),
            },
        )

    st.markdown("**Who paid for what**")
    matrix = (exp.pivot_table(index="Category", columns="Paid by", values="_paise",
                              aggfunc="sum", fill_value=0)
              .reindex(columns=names, fill_value=0) / 100)
    matrix["Total"] = matrix.sum(axis=1)
    matrix.loc["Total"] = matrix.sum()
    st.dataframe(matrix.style.format("{:,.2f}"), use_container_width=True)

    st.markdown("### 🔁 Final settlement")
    transfers = settle_up(balances)
    if len(names) > 1:
        if transfers:
            for debtor, creditor, amount in transfers:
                st.markdown(f"- **{debtor}** pays **{creditor}** {money(amount)}")
        else:
            st.success("Everyone has paid exactly their share. Nothing to settle.")
    st.session_state["trip_budget_summary"] = {
        "group": group, "method": method, "days": int(b_days), "total": total_paise / 100,
        "by_cat": [(r["Category"], float(r[AMT])) for _, r in by_cat.iterrows()],
        "result": result.to_dict("records"),
        "transfers": [(d, c, a / 100) for d, c, a in transfers],
    }


def page_budget():
    render_trip_budget()

# =====================================================
# 6. ITINERARY PLANNER & TSP ROUTING
# =====================================================
# Major India-wide holidays and festivals. Moon-based dates are indicative;
# regional/state holiday calendars can differ.
INDIA_HOLIDAYS = {
    2026: {
        "01-26": ("Republic Day", "National holiday"),
        "03-04": ("Holi", "Festival"),
        "03-21": ("Eid-ul-Fitr", "Festival · moon sighting may affect date"),
        "03-26": ("Ram Navami", "Festival"),
        "03-31": ("Mahavir Jayanti", "Festival"),
        "04-03": ("Good Friday", "Festival / public holiday"),
        "05-01": ("Buddha Purnima", "Festival"),
        "05-27": ("Eid-ul-Zuha (Bakrid)", "Festival · moon sighting may affect date"),
        "06-26": ("Muharram", "Festival · moon sighting may affect date"),
        "08-15": ("Independence Day", "National holiday"),
        "08-26": ("Milad-un-Nabi (Id-e-Milad)", "Festival · moon sighting may affect date"),
        "09-04": ("Janmashtami", "Festival"),
        "10-02": ("Gandhi Jayanti", "National holiday"),
        "10-19": ("Durga Puja / Maha Navami (regional)", "Regional festival · dates vary by state"),
        "10-20": ("Dussehra / Vijayadashami", "Festival"),
        "11-08": ("Diwali (Deepavali)", "Festival"),
        "11-24": ("Guru Nanak Jayanti", "Festival"),
        "12-25": ("Christmas Day", "Festival / public holiday"),
    },
    2027: {
        "01-15": ("Makar Sankranti / Pongal", "Festival"),
        "01-26": ("Republic Day", "National holiday"),
        "02-11": ("Vasant Panchami", "Festival"),
        "03-06": ("Maha Shivaratri", "Festival"),
        "03-10": ("Eid-ul-Fitr (expected)", "Festival · tentative, moon sighting may affect date"),
        "03-22": ("Holi / Dolyatra", "Festival"),
        "03-26": ("Good Friday", "Festival / public holiday"),
        "04-07": ("Ugadi / Gudi Padwa", "Festival"),
        "04-14": ("Baisakhi / Vishu", "Festival"),
        "08-15": ("Independence Day", "National holiday"),
        "08-17": ("Raksha Bandhan", "Festival"),
        "08-25": ("Janmashtami", "Festival"),
        "09-04": ("Ganesh Chaturthi", "Festival"),
        "09-12": ("Onam", "Festival"),
        "10-05": ("Durga Puja begins (regional)", "Regional festival · dates vary by state"),
        "10-06": ("Durga Puja · Maha Saptami", "Regional festival"),
        "10-07": ("Durga Puja · Maha Ashtami", "Regional festival"),
        "10-08": ("Durga Puja · Maha Navami", "Regional festival"),
        "10-09": ("Dussehra / Vijayadashami", "Festival"),
        "10-29": ("Diwali (Deepavali)", "Festival"),
        "11-04": ("Chhath Puja", "Festival · especially observed in eastern India"),
        "11-14": ("Guru Nanak Jayanti", "Festival"),
        "12-25": ("Christmas Day", "Festival / public holiday"),
    },
}


for _y in (2026, 2027, 2028):
    INDIA_HOLIDAYS.setdefault(_y, {}).setdefault("01-01", ("New Year's Day", "Public holiday"))
INDIA_HOLIDAYS[2026].setdefault("01-14", ("Makar Sankranti / Pongal", "Festival"))
for _md, _v in {"01-26": ("Republic Day", "National holiday"), "08-15": ("Independence Day", "National holiday"),
                "10-02": ("Gandhi Jayanti", "National holiday"), "12-25": ("Christmas Day", "Festival / public holiday")}.items():
    INDIA_HOLIDAYS[2028].setdefault(_md, _v)


def holidays_for_date(date_value):
    """Return known major Indian holidays for a date, or an empty list."""
    if not isinstance(date_value, (datetime.date, datetime.datetime)):
        return []
    entry = INDIA_HOLIDAYS.get(date_value.year, {}).get(date_value.strftime("%m-%d"))
    return [entry] if entry else []


def show_holiday_note(date_value, prefix="📅"):
    """Show a friendly holiday indicator for the selected itinerary date."""
    matches = holidays_for_date(date_value)
    if matches:
        for holiday_name, holiday_type in matches:
            st.success(f"{prefix} **{holiday_name}** — {holiday_type}")
    elif date_value.year not in INDIA_HOLIDAYS:
        st.caption("Holiday lookup is currently available for 2026–2027. Check your state holiday calendar for other years.")
    else:
        st.caption("No major India-wide holiday is listed for this date. State or local holidays may still apply.")


def _auto_pick_stops(city, days, limit_h):
    """Fill the stop list with the best-rated places that fit the trip length."""
    pool = active_df[active_df["_city"] == city].copy()
    if pool.empty:
        return
    base = pool["_rating"].median() if pool["_rating"].notna().any() else 4.0
    pool["_pick_score"] = (
        pool["_rating"].fillna(base) + 0.3 * np.log1p(pool["_reviews"].fillna(0))
        + np.where(pool["_latitude"].notna(), 0.4, 0.0)
    )
    pool = pool.sort_values("_pick_score", ascending=False)
    budget_h, used, picked = max(int(days), 1) * float(limit_h) * 0.75, 0.0, []
    for _, r in pool.iterrows():
        h = r.get("_hours_needed")
        h = (float(h) if pd.notna(h) and float(h) > 0 else visit_hours(r["_type"], r["_name"])) + 0.5
        if picked and used + h > budget_h:
            continue
        picked.append(r["_name"])
        used += h
        if len(picked) >= 12:
            break
    if len(picked) < 2:
        picked = pool["_name"].head(2).tolist()
    st.session_state[f"itin_stops_{city}"] = picked


def page_itinerary():
    st.header("🗓️ Smart Itinerary: day clusters, weather and budget link")
    city_options = sorted(c for c in active_df["_city"].dropna().unique() if c)
    if not city_options:
        st.info("No cities available in the current selection.")
    else:
        i1, i2, i3 = st.columns(3)
        with i1:
            keep("itin_city", city_options[0], city_options)
            selected_itin_city = st.selectbox("City", city_options, key="itin_city")
        with i2:
            keep("itin_travelers", 2)
            itin_travelers = st.number_input("Travellers", 1, 15, key="itin_travelers")
        with i3:
            _today = datetime.date.today()
            _max_trip_date = _today + datetime.timedelta(days=365 * 5)
            _default_start = _today + datetime.timedelta(days=1)
            _default_end = _default_start + datetime.timedelta(days=1)

            # Keep the two calendar selections independent so users can
            # browse and choose each date freely in the normal calendar picker.
            if not isinstance(st.session_state.get("itin_start_date"), datetime.date):
                st.session_state["itin_start_date"] = _default_start
            if not isinstance(st.session_state.get("itin_return_date"), datetime.date):
                st.session_state["itin_return_date"] = _default_end

            itin_date = st.date_input(
                "Visit / trip start date",
                min_value=_today,
                max_value=_max_trip_date,
                format="DD/MM/YYYY",
                key="itin_start_date",
                help="Open the calendar and use the arrows to browse months. Choose any available future date.",
            )
            _trip_end_date = st.date_input(
                "Return date",
                min_value=_today,
                max_value=_max_trip_date,
                format="DD/MM/YYYY",
                key="itin_return_date",
                help="Choose your return date from the calendar. It must be on or after your start date.",
            )

        if isinstance(itin_date, datetime.date) and isinstance(_trip_end_date, datetime.date):
            if _trip_end_date < itin_date:
                itin_days = 0
                st.warning("Your return date is before your start date. Please choose a later return date.")
            else:
                itin_days = (_trip_end_date - itin_date).days + 1
                st.caption(f"🗓️ Trip duration: **{itin_days} day(s)** · {itin_date.strftime('%d %b %Y')} to {_trip_end_date.strftime('%d %b %Y')}")
                show_holiday_note(itin_date, "🎉 Start date")
                if _trip_end_date != itin_date:
                    show_holiday_note(_trip_end_date, "🏁 Return date")
                if itin_days > 14:
                    st.warning("The itinerary currently supports trips up to 14 days. Please select a shorter date range.")
        else:
            itin_date = None
            itin_days = 0
            st.info("Choose your trip start date and return date using the calendars.")
        i4, i5 = st.columns(2)
        with i4:
            keep("itin_start", 9)
            itin_start = st.slider("Day starts at (hour)", 6, 12, key="itin_start")
        with i5:
            keep("itin_limit", 8)
            itin_limit = st.slider("Max sightseeing hours per day", 4, 12, key="itin_limit")

        city_places = active_df[active_df["_city"] == selected_itin_city]
        _stop_options = city_places["_name"].tolist()
        keep(f"itin_stops_{selected_itin_city}", _stop_options[:min(4, len(_stop_options))], _stop_options, multi=True)
        st.button(
            "✨ Suggest stops for me", key="itin_autopick", on_click=_auto_pick_stops,
            args=(selected_itin_city, itin_days, itin_limit),
            help="Picks the best-rated places in this city that fit your trip length and daily hours.",
        )
        selected_stops = st.multiselect(
            "Select 2 to 12 stops", _stop_options, key=f"itin_stops_{selected_itin_city}",
        )

        if itin_date is None or itin_days == 0:
            pass
        elif itin_days > 14:
            pass
        elif not 2 <= len(selected_stops) <= 12:
            st.info("Choose between 2 and 12 stops to build a plan.")
        else:
            subset_df = city_places[city_places["_name"].isin(selected_stops)].reset_index(drop=True)
            subset_df["_xy_approx"] = False
            _miss = subset_df["_latitude"].isna() | subset_df["_longitude"].isna()
            if _miss.any():
                _known = city_places.dropna(subset=["_latitude", "_longitude"])
                _ref = (float(_known["_latitude"].median()), float(_known["_longitude"].median())) if len(_known) else None
                with st.spinner(f"Finding locations for {int(_miss.sum())} place(s) without coordinates..."):
                    for _i in subset_df.index[_miss]:
                        _r = subset_df.loc[_i]
                        _xy, _approx = geocode_cached(_r["_name"], _r["_city"], _r["_state"]), False
                        if _xy and _ref and haversine_km(_ref, _xy) > 80:
                            _xy = None  # geocoder picked a same-named place in another region
                        if not _xy:
                            _xy, _approx = (_ref, True) if _ref else (geocode_cached(_r["_city"], "", _r["_state"]), True)
                        if _xy:
                            subset_df.loc[_i, ["_latitude", "_longitude"]] = [float(_xy[0]), float(_xy[1])]
                            subset_df.loc[_i, "_xy_approx"] = _approx
                            if not _approx:
                                save_geo_fill(_r["_name"], _r["_city"], _r["_state"], _xy[0], _xy[1])
            no_xy = subset_df[subset_df["_latitude"].isna() | subset_df["_longitude"].isna()]
            if len(no_xy):
                st.warning("No GPS coordinates for: " + ", ".join(no_xy["_name"]) + ". They are left out of the route.")
            plan = build_day_plan(subset_df, int(itin_days), float(itin_start), float(itin_limit), start_date=itin_date)
            if plan:
                plan = render_plan_editor(plan, subset_df, float(itin_start), float(itin_limit), selected_itin_city)
            if not plan:
                st.warning("Need at least one stop with coordinates.")
            else:
                if len(plan) < itin_days:
                    st.info(f"Only {len(plan)} day(s) filled, because there are fewer stops than days.")
                summary_days, total_fee, total_km, unlisted = [], 0.0, 0.0, 0
                for day in plan:
                    date_txt = day["date"].strftime("%a, %d %b") if day["date"] else ""
                    with st.container(border=True):
                        st.markdown(f"### Day {day['day']}  ·  {date_txt}")
                        for holiday_name, holiday_type in holidays_for_date(day["date"]):
                            st.success(f"🎉 Holiday / festival: **{holiday_name}** — {holiday_type}")
                        _ratio = (day["end"] - float(itin_start)) / max(float(itin_limit), 1.0)
                        _pace = "😌 Relaxed" if _ratio < 0.6 else ("⚖️ Balanced" if _ratio <= 1.0 else "🔥 Packed")
                        _day_fee = sum(x["fee"] for x in day["stops"] if x["fee"] is not None)
                        st.caption(f"{_pace} · {len(day['stops'])} stops · about {day['km']:.0f} km of local travel · finishes around {fmt_clock(day['end'])} · entry fees ₹{_day_fee:,.0f} per person")
                        if day["overloaded"]:
                            st.warning(f"This day runs past your {itin_limit}-hour limit. Add a day or remove a stop.")
                        w = weather_for_day(*day["centroid"], day["date"]) if day["date"] else None
                        advice, w_text = [], ""
                        if w:
                            w_text = weather_line(w)
                            advice = weather_advice(w, day["stops"])
                            st.markdown(f"🌤️ **Forecast:** {w_text}")
                            for tip in advice:
                                st.warning(tip)
                            if not advice:
                                st.success("Weather looks comfortable for sightseeing.")
                        else:
                            st.caption("🌤️ Forecast is available for the next 16 days only.")
                        if not day["stops"]:
                            st.info("🌴 Free day: nothing planned.")
                        for n, s in enumerate(day["stops"], start=1):
                            if s["lunch_before"]:
                                st.caption("🍽️ Lunch break (1 h)")
                            fee_txt = "fee not listed" if s["fee"] is None else f"₹{s['fee']:,.0f}"
                            st.markdown(
                                f"**{n}. {s['name']}** · {s['type']}  \n"
                                f"{fmt_clock(s['arrive'])} to {fmt_clock(s['leave'])} (about {s['hours']:g} h) · {fee_txt}"
                                + (f" · {s['travel_km']:.1f} km from previous stop" if s["travel_km"] else "")
                            )
                            _extra = []
                            if s.get("slot"):
                                _extra.append(f"🕒 Best in the {s['slot']}" + ("" if s["slot_ok"] else f" (this plan arrives {fmt_clock(s['arrive'])})"))
                            if s.get("transport"):
                                _extra.append(f"🚉 {s['transport']}" + ("" if s["transport_verified"] else " · estimate"))
                            if s.get("approx_xy"):
                                _extra.append("📍 exact location not found, placed at city centre")
                            if _extra:
                                st.caption("  ·  ".join(_extra))
                            for note in s["notes"]:
                                st.caption(f"⚠️ {note}")
                            if s.get("user_note"):
                                st.caption(f"📝 {s['user_note']}")
                            if s["fee"] is None:
                                unlisted += 1
                            else:
                                total_fee += s["fee"]
                        total_km += day["km"]
                    summary_days.append({
                        "day": day["day"], "date": date_txt, "weather": w_text, "advice": advice,
                        "stops": [{
                            "name": s["name"], "type": s["type"], "fee": s["fee"],
                            "arrive": fmt_clock(s["arrive"]), "leave": fmt_clock(s["leave"]), "notes": s["notes"],
                        } for s in day["stops"]],
                    })

                st.subheader("🗺️ Day-wise route map")
                st.caption("Each colour is one day; days are grouped by location so you don't criss-cross the city.")
                render_day_plan_map(plan, key="itin_day_map")

                st.subheader("💰 Budget link")
                b1, b2, b3 = st.columns(3)
                with b1: render_metric_card("Entry fees / person", f"₹{total_fee:,.0f}", "🎟️")
                with b2: render_metric_card("Entry fees / group", f"₹{total_fee * itin_travelers:,.0f}", "👥")
                with b3: render_metric_card("Local travel", f"{total_km:.0f} km", "🚗")
                if unlisted:
                    st.caption(f"{unlisted} stop(s) have no listed fee and count as ₹0 here.")
                if st.button("💰 Send to Travel Budget", type="primary", key="itin_send_budget"):
                    st.session_state["itin_budget"] = {
                        "days": len(plan), "entry_pp": float(total_fee), "unlisted": unlisted,
                        "km": float(total_km), "stops": [s["name"] for d in plan for s in d["stops"]],
                        "city": selected_itin_city,
                    }
                    st.success("Sent. Open Plan My Trip, choose Budget and press 'Use itinerary data in this budget'.")
                st.session_state["trip_itinerary_summary"] = {
                    "city": selected_itin_city, "travelers": int(itin_travelers), "days": summary_days,
                }
                if st.button("💾 Save this trip", key="itin_save_trip"):
                    _prof = st.session_state["profile"]
                    _title = f"{selected_itin_city} · {itin_date.strftime('%d %b %Y')}"
                    _trip = {"title": _title, "city": selected_itin_city, "travelers": int(itin_travelers),
                             "start": itin_date.isoformat(), "end": _trip_end_date.isoformat(),
                             "stops": list(selected_stops), "edits": st.session_state.get("ed_current", {}),
                             "saved": now_ist().strftime("%d %b %Y")}
                    _prof["saved_trips"] = [t for t in _prof.get("saved_trips", []) if t.get("title") != _title][-11:] + [_trip]
                    profile_save()
                    st.success("Trip saved. Find it under My Trip → Saved trips."
                               + (" (Guest trips are lost when you close the tab.)" if st.session_state["user"]["guest"] else ""))

    with st.expander("📅 Trip calendar, holidays & special days", expanded=False):
        page_calendar()


# =====================================================
# 7. DESTINATION DETAILS
# =====================================================
def page_details():
    st.header("🧭 Destination Details")
    all_names = active_df["_name"].unique().tolist()
    if all_names:
        keep("detail_select", all_names[0], all_names)
        chosen_place = st.selectbox("Choose a destination", all_names, key="detail_select")
        p_row = active_df[active_df["_name"] == chosen_place].iloc[0]
        show_destination(p_row, tab_name="details_tab")
        with st.expander("🗺️ Location map", expanded=True):
            render_explorer_map(pd.DataFrame([p_row]), key="details_map", height=360)
            if st.session_state.get("_map_approx"):
                st.caption("Approximate location: exact coordinates are not in the dataset yet.")
        with st.container(border=True):
            page_transport(p_row)

# =====================================================
# 8. TRANSPORT & CONNECTIVITY
# =====================================================
def _fmt_hours(h):
    mins = int(round(h * 60))
    hh, mm = divmod(mins, 60)
    return f"{hh} h {mm:02d} m" if hh else f"{mm} m"


def journey_options(straight_km, travellers=1):
    """Rough one-way options from straight-line distance. Estimates only."""
    n = max(1, int(travellers))
    cars = math.ceil(n / 4)
    road = straight_km * 1.3
    rail = straight_km * 1.2
    bus_pp = road * TRANSPORT_RATES["Bus (road-distance estimate)"]
    train_pp = rail * TRANSPORT_RATES["Train (approximate)"]
    car_each = road * TRANSPORT_RATES["Car"]
    fly_pp = straight_km * TRANSPORT_RATES["Flight (approximate)"]

    def pp_note(pp, extra=""):
        base = "per person" if n == 1 else f"₹{pp:,.0f} per person × {n}"
        return base + extra

    opts = [
        {"mode": "Bus", "icon": "🚌", "km": road, "hours": road / 40, "cost": bus_pp * n, "note": pp_note(bus_pp)},
        {"mode": "Train", "icon": "🚆", "km": rail, "hours": rail / 55 + 0.5, "cost": train_pp * n, "note": pp_note(train_pp)},
        {"mode": "Car", "icon": "🚗", "km": road, "hours": road / 45, "cost": car_each * cars,
         "note": f"{cars} car(s), ₹{car_each:,.0f} each (4 seats per car)"},
    ]
    if straight_km >= 400:
        opts.append({"mode": "Flight", "icon": "✈️", "km": straight_km, "hours": straight_km / 650 + 3.0,
                     "cost": fly_pp * n, "note": pp_note(fly_pp, ", incl. ~3 h airport time")})
    best = "Bus" if straight_km < 300 else ("Train" if straight_km <= 1000 else "Flight")
    return opts, best


def _collapse_airport_terminals(frame):
    """Merge 'Airport Arrival Terminal 2', 'Old Terminal' etc. into one airport (nearest kept)."""
    if frame is None or frame.empty or "Category" not in frame.columns:
        return frame
    is_air = frame["Category"] == "Airport"
    if not is_air.any():
        return frame
    air = frame[is_air].copy()
    air["Name"] = air["Name"].str.replace(
        r"\s*[-,]?\s*((arrival|departure|old|new|domestic|international|cargo)\s+)*terminal\s*[\w-]*\s*$|\s+(arrival|departure)s?\s*$",
        "", regex=True, flags=re.IGNORECASE).str.strip()
    air = air.sort_values("Distance (km)", na_position="last").drop_duplicates(subset=["Name"])
    return pd.concat([frame[~is_air], air], ignore_index=True)


def render_journey_section(start_city, dest_name, dest_lat, dest_lon):
    st.markdown(f"### 🧳 Getting to {dest_name} from {start_city or 'your city'}")
    if not str(start_city).strip():
        st.info("Enter your starting city above to see travel options.")
        return
    start_xy = geocode_destination(str(start_city).strip())
    maps_url = (f"https://www.google.com/maps/dir/?api=1&origin={quote_plus(str(start_city).strip())}"
                f"&destination={dest_lat},{dest_lon}&travelmode=transit")
    if not start_xy:
        st.warning("I could not locate your starting city. Try a bigger nearby city name.")
        st.link_button("Open route in Google Maps", maps_url)
        return
    straight = haversine_km(start_xy, (dest_lat, dest_lon))
    if straight < 5:
        st.success("You are already within a few km of this destination. Use local transport below.")
        return
    st.session_state.setdefault("trans_travellers", 1)
    tcol, _ = st.columns([1, 3])
    travellers = tcol.number_input("Travellers", min_value=1, max_value=30, key="trans_travellers")
    opts, best = journey_options(straight, travellers)
    st.session_state["trip_route"] = {
        "start": str(start_city).strip(), "dest": str(dest_name), "straight_km": float(straight),
        "best": best, "travellers": int(travellers),
        "options": [{"mode": o["mode"], "time": _fmt_hours(o["hours"]), "cost": float(o["cost"]), "note": o["note"]} for o in opts],
    }
    st.caption(f"Straight-line distance about {straight:,.0f} km (one way). Costs are totals for {int(travellers)} traveller(s).")
    cols = st.columns(len(opts))
    for col, o in zip(cols, opts):
        is_best = o["mode"] == best
        border = "#71834A" if is_best else "#DCE3D1"
        badge = ("<div style='color:#354522;background:#E3EBD3;border-radius:999px;display:inline-block;"
                 "padding:1px 10px;font-size:.75rem;font-weight:600;margin-bottom:4px'>✅ Best fit</div>") if is_best else ""
        col.markdown(
            f"<div style='border:2px solid {border};border-radius:14px;padding:12px;background:#fff;text-align:center'>"
            f"{badge}<div style='font-size:1.6rem'>{o['icon']}</div>"
            f"<div style='font-weight:700;color:#27311F'>{o['mode']}</div>"
            f"<div style='color:#354522;font-size:1.05rem;font-weight:600;margin-top:4px'>{_fmt_hours(o['hours'])}</div>"
            f"<div style='color:#27311F'>≈ ₹{o['cost']:,.0f}</div>"
            f"<div style='color:#59634D;font-size:.75rem'>{html.escape(o['note'])}</div></div>",
            unsafe_allow_html=True,
        )
    st.caption("Estimates only (road ≈ 1.3× and rail ≈ 1.2× straight-line). Check IRCTC, bus or airline sites for real timings and fares.")
    st.link_button("🧭 Open route in Google Maps", maps_url)


def transport_place_column(category, subset, dest_lat, dest_lon, start_city=""):
    icon = TRANSPORT_ICONS.get(category, "📍")
    st.markdown(
        f"<div style='font-weight:700;color:#354522;font-size:1.05rem;margin:6px 0 8px'>{icon} {html.escape(category)}</div>",
        unsafe_allow_html=True)
    if subset.empty:
        st.caption("No result returned.")
        return
    start_city = str(start_city or "").strip()
    for _, item in subset.iterrows():
        name = str(item["Name"])
        detail = str(item.get("Detail") or "")
        dist = f"{item['Distance (km)']:.1f} km" if pd.notna(item["Distance (km)"]) else "–"
        if str(item.get("Source", "")) == "OpenStreetMap" and detail.lower() in {"station", "halt", "subway", "light_rail", "monorail"}:
            detail = ""
        _la, _lo = item.get("lat"), item.get("lon")
        if _la is not None and _lo is not None and pd.notna(_la) and pd.notna(_lo):
            dest_text = f"{float(_la)},{float(_lo)}"
        else:
            dest_text = quote_plus(f"{name}, {detail}".strip(", "))
        base = "https://www.google.com/maps/dir/?api=1&travelmode=transit&destination=" + dest_text
        links = [f"<a href='{html.escape(base + f'&origin={dest_lat},{dest_lon}')}' target='_blank'>📍 From destination</a>"]
        if start_city:
            links.append(f"<a href='{html.escape(base + '&origin=' + quote_plus(start_city))}' target='_blank'>🧭 From {html.escape(start_city)}</a>")
        st.markdown(
            "<div style='border:1px solid #DCE3D1;border-radius:12px;padding:10px 12px;margin-bottom:8px;background:#fff'>"
            "<div style='display:flex;justify-content:space-between;gap:8px;align-items:flex-start'>"
            f"<div style='font-weight:600;color:#27311F'>{html.escape(name)}</div>"
            f"<div style='background:#EEF2E4;color:#354522;border-radius:999px;padding:2px 10px;font-size:.8rem;white-space:nowrap'>{dist}</div></div>"
            f"<div style='color:#59634D;font-size:.8rem;margin:3px 0 6px'>{html.escape(detail)}</div>"
            f"<div style='font-size:.82rem'>{' · '.join(links)}</div></div>",
            unsafe_allow_html=True)


def page_transport(selected_row=None):
    if selected_row is None:
        st.header("🚆 Transport & connectivity")
        st.caption("Nearest railway stations, metro, bus stops and airports, plus how to reach the destination from your starting city.")
    else:
        st.markdown("### 🚆 Transport & connectivity")
        st.caption("Nearest railway stations, metro, bus stops and airports, plus estimated travel options from your starting city.")
    _tp = active_df.dropna(subset=["_name"]).drop_duplicates(subset=["_name", "_city"])
    _name_counts = _tp["_name"].value_counts()
    rows_by_label = {}
    for _, _r in _tp.iterrows():
        _c = "" if pd.isna(_r.get("_city")) else str(_r.get("_city")).strip()
        _lab = f"{_r['_name']} — {_c}" if (_name_counts[_r["_name"]] > 1 and _c) else str(_r["_name"])
        rows_by_label.setdefault(_lab, _r)
    trans_places = list(rows_by_label)
    if not trans_places:
        st.info("No destinations are available in the current selection.")
        return

    t1, t0, t2 = st.columns([3, 2, 1])
    with t0:
        keep("ai_start_city", st.session_state.get("profile", {}).get("home_city") or "Kolkata")
        start_city = st.text_input("Your starting city", key="ai_start_city")
    if selected_row is None:
        with t1:
            keep("trans_place_select", trans_places[0], trans_places)
            selected_trans_place = st.selectbox(
                "Select destination to check connectivity", trans_places, key="trans_place_select"
            )
        t_row = rows_by_label[selected_trans_place]
    else:
        t_row = selected_row
        with t1:
            st.text_input("Selected destination", value=f"{t_row['_name']} · {t_row.get('_city', '')}", disabled=True)
    with t2:
        keep_range("trans_radius", 10, 5, 10)
        t_radius = st.slider("Search radius (km)", 5, 10, key="trans_radius")

    t_xy = (t_row["_latitude"], t_row["_longitude"])
    coords_ok = False
    try:
        coords_ok = (
            pd.notna(t_xy[0]) and pd.notna(t_xy[1])
            and 6 <= float(t_xy[0]) <= 38 and 68 <= float(t_xy[1]) <= 98
        )
    except (TypeError, ValueError):
        coords_ok = False

    if not coords_ok:
        with st.spinner("Finding the destination's map coordinates..."):
            t_xy = geocode_cached(t_row["_name"], t_row["_city"], t_row["_state"])
        if t_xy and 6 <= float(t_xy[0]) <= 38 and 68 <= float(t_xy[1]) <= 98:
            save_geo_fill(t_row["_name"], t_row["_city"], t_row["_state"], t_xy[0], t_xy[1])
    if not t_xy:
        st.warning("I could not find valid map coordinates for this destination. Transport results need valid coordinates.")
        dest_lat = dest_lon = None
    else:
        dest_lat, dest_lon = float(t_xy[0]), float(t_xy[1])
        if not (6 <= dest_lat <= 38 and 68 <= dest_lon <= 98):
            st.warning("The destination coordinates do not appear to be in India. Please correct the latitude/longitude in the dataset.")
            dest_lat = dest_lon = None

    place_query = ", ".join(str(x) for x in (t_row["_name"], t_row.get("_city", ""), t_row.get("_state", "")) if str(x).strip())
    if dest_lat is not None and dest_lon is not None:
        render_journey_section(start_city, t_row["_name"], dest_lat, dest_lon)
        st.divider()
    st.markdown(f"### 🧭 Transport near {t_row['_name']}")
    st.caption(f"Showing the nearest places within about {t_radius} km (the search service allows at most 10 km). If no airport is that close, the nearest one is looked up separately.")

    render_dataset_transport(t_row)
    if dest_lat is None or dest_lon is None:
        st.warning("Valid destination coordinates are required for the live nearby search.")
        return

    results, source, err = None, "", ""
    mappls_key = get_mappls_static_key()
    if mappls_key:
        try:
            with st.spinner("Searching Mappls for nearby railway, metro, bus and airport facilities..."):
                found = fetch_mappls_transport(dest_lat, dest_lon, t_radius, mappls_key)
            if found is not None and not found.empty:
                results, source = found, "Mappls"
        except RuntimeError as exc:
            err = str(exc)
        except Exception as exc:
            err = f"Mappls search failed: {type(exc).__name__}."
        if results is not None and "Airport" not in set(results["Category"]):
            try:
                osm_all = fetch_transport(round(dest_lat, 5), round(dest_lon, 5), t_radius)
                osm_air = osm_all[osm_all["Category"] == "Airport"].head(3)
                if not osm_air.empty:
                    extra = pd.DataFrame({
                        "Category": "Airport", "Name": osm_air["Name"],
                        "Distance (km)": osm_air["Distance (km)"],
                        "Detail": "From OpenStreetMap (straight-line); Mappls radius is capped at 10 km",
                        "eLoc": "", "Source": "OpenStreetMap",
                    })
                    results = pd.concat([results, extra], ignore_index=True)
            except Exception:
                pass
    if results is None:
        try:
            with st.spinner("Checking OpenStreetMap for mapped transport facilities..."):
                found = fetch_transport(round(dest_lat, 5), round(dest_lon, 5), t_radius)
            if found is not None and not found.empty:
                results, source = found, "OpenStreetMap"
        except RuntimeError:
            err = err or "OpenStreetMap is currently unavailable. Please try again shortly."
    if results is None:
        st.info(err or "No named transport places were found within this radius. Try a larger radius.")
        if not mappls_key:
            st.caption("Adding a MAPPLS_API_KEY (see the README) gives more complete results.")
        return

    results = _collapse_airport_terminals(results)
    st.success(f"Found {len(results)} nearby places from {source}.")
    nearest_rows = []
    columns = st.columns(2)
    for idx, category in enumerate(TRANSPORT_ORDER):
        subset = results[results["Category"] == category].sort_values("Distance (km)", na_position="last").head(3)
        if not subset.empty:
            nearest_rows.extend(subset.to_dict("records"))
        with columns[idx % 2]:
            transport_place_column(category, subset, dest_lat, dest_lon, start_city)
    if source == "OpenStreetMap" and nearest_rows:
        st.markdown("### 📍 Results map")
        m = folium.Map(location=[dest_lat, dest_lon], zoom_start=12, tiles="OpenStreetMap")
        folium.Marker([dest_lat, dest_lon], tooltip=f"Destination: {t_row['_name']}", icon=folium.Icon(color="green", icon="star")).add_to(m)
        color_by_category = {"Railway station": "blue", "Metro": "purple", "Bus": "orange", "Airport": "red"}
        for item in nearest_rows:
            if item.get("lat") is None or item.get("lon") is None:
                continue
            folium.Marker(
                [float(item["lat"]), float(item["lon"])],
                tooltip=f"{item['Category']}: {item['Name']}",
                icon=folium.Icon(color=color_by_category.get(item["Category"], "gray"), icon="info-sign"),
            ).add_to(m)
        st_folium(m, width="100%", height=430, returned_objects=[], key="transport_map_backup")
    if source == "Mappls":
        st.caption("Source: Mappls Nearby API. Distances are supplied by Mappls; confirm station details and transit services before travelling.")
    else:
        st.caption("Source: OpenStreetMap contributors via Overpass API. Distances are straight-line, not road distances; verify schedules with the operator.")


# =====================================================
# 9. FAVORITES
# =====================================================
def page_favorites():
    st.header("⭐ Saved Favorites")
    favs = st.session_state["favorites"]
    if not favs:
        st.info("No saved favorites yet! Click 'Add to Favorites' on any destination card.")
    else:
        fav_ids = set(favs)
        fav_df = df[df.apply(lambda r: f"{r['_name']}_{r['_city']}" in fav_ids, axis=1)]
        st.button("🗑️ Clear all favorites", key="fav_clear_all", on_click=clear_favorites)
        for _, row in fav_df.iterrows():
            show_destination(row, tab_name="favs")




# =====================================================
# 12. TRIP SUMMARY (PDF / Markdown export)
# =====================================================
def page_summary():
    st.header("📄 Trip Summary")
    st.write("Combines your itinerary, budget and favourites into one document.")
    trip_title = st.text_input("Trip title", value=f"My trip to {selected_global_state}", key="sum_title")
    favs_ids = set(st.session_state["favorites"])
    fav_names = [
        f"{r['_name']} ({r['_city']}, {r['_state']})" if r["_city"] else f"{r['_name']} ({r['_state']})"
        for _, r in df.iterrows() if f"{r['_name']}_{r['_city']}" in favs_ids
    ]
    summary_ctx = {
        "title": trip_title or "My trip", "focus": selected_global_state,
        "itinerary": st.session_state.get("trip_itinerary_summary"),
        "budget": st.session_state.get("trip_budget_summary"),
        "transport": st.session_state.get("trip_route"),
        "favorites": fav_names,
    }
    s1, s2, s3 = st.columns(3)
    s1.metric("Itinerary", "Ready" if summary_ctx["itinerary"] else "Not built")
    s2.metric("Budget", "Ready" if summary_ctx["budget"] else "Not calculated")
    s3.metric("Favourites", len(fav_names))
    if not summary_ctx["itinerary"] or not summary_ctx["budget"]:
        st.info("Build the itinerary in the Itinerary Planner tab and finish the Travel Budget calculation to fill the missing sections.")

    md_text = trip_summary_markdown(summary_ctx)
    _pk = st.session_state.get("trip_packing_summary")
    if _pk:
        md_text += f"\n\n## Packing checklist\nPacked {_pk['packed']} of {_pk['total']} items.\n"
        md_text += "\n".join(f"- [ ] {x}" for x in _pk["remaining"])
    with st.expander("Preview", expanded=True):
        st.markdown(md_text)
    d1, d2 = st.columns(2)
    pdf_bytes = build_trip_pdf(summary_ctx)
    with d1:
        if pdf_bytes:
            st.download_button("⬇️ Download PDF", pdf_bytes, file_name="trip_summary.pdf", mime="application/pdf", type="primary", key="sum_pdf")
        else:
            st.button("⬇️ Download PDF", disabled=True, key="sum_pdf_off")
            st.caption("PDF export needs the `reportlab` package: `pip install reportlab`.")
    with d2:
        st.download_button("⬇️ Download Markdown", md_text, file_name="trip_summary.md", mime="text/markdown", key="sum_md")





# =====================================================
# NAVIGATION (sidebar, grouped)
# =====================================================
EXPLORE_SECTIONS = ["📍 Explore places", "🧭 Place details"]
PLAN_SECTIONS = ["🗓️ Itinerary", "🎒 Packing", "💰 Budget"]
MYTRIP_SECTIONS = ["⭐ Favorites", "🧳 Saved trips", "👤 Profile", "📄 Trip Summary"]


def _section_radio(key, options):
    keep(key, options[0], options)
    return st.radio("Section", options, horizontal=True, key=key, label_visibility="collapsed")


# ---------- Itinerary editing ----------
def _hav(a, b):
    return haversine_km((a["lat"], a["lon"]), (b["lat"], b["lon"]))


def _insert_cheapest(seq, s):
    if not seq:
        return [s]
    best = (1e18, len(seq))
    for i in range(len(seq) + 1):
        prev, nxt = (seq[i - 1] if i > 0 else None), (seq[i] if i < len(seq) else None)
        add = _hav(prev, s) + _hav(s, nxt) - _hav(prev, nxt) if prev and nxt else _hav(prev or nxt, s)
        best = min(best, (add, i))
    return seq[:best[1]] + [s] + seq[best[1]:]


def apply_plan_edits(plan, edits, rows, start_hour, limit_h, speed_kmh=25.0):
    nd = len(plan)
    home, moved = [[] for _ in range(nd)], [[] for _ in range(nd)]
    for d in plan:
        for s in d["stops"]:
            e = edits.get(s["name"], {})
            if e.get("skip"):
                continue
            s = dict(s, hours=float(e.get("hours", s["hours"])), user_note=e.get("note", ""))
            tgt = min(max(int(e.get("day", d["day"])), 1), nd) - 1
            (home if tgt == d["day"] - 1 else moved)[tgt].append(s)
    out = []
    for d, base, extra in zip(plan, home, moved):
        seq = base
        for s in extra:
            seq = _insert_cheapest(seq, s)
        t, prev, km_total, lunch_done, new = start_hour, None, 0.0, False, []
        for s in seq:
            notes, travel = [], 0.0
            if prev is not None:
                travel = _hav(prev, s) * 1.3
                t += travel / speed_kmh
                km_total += travel
                if travel / speed_kmh > 2:
                    notes.append(f"Long transfer of about {travel / speed_kmh:.1f} h from the previous stop; consider a separate day")
            row = rows.get(s["name"])
            open_h, close_h = place_hours(row) if row is not None else (None, None)
            if open_h is not None and t < open_h:
                notes.append(f"Opens at {fmt_clock(open_h)}, so arrival is moved to opening time")
                t = open_h
            lunch = False
            if not lunch_done and t >= 13:
                t, lunch_done, lunch = t + 1.0, True, True
            arrive, leave = t, t + s["hours"]
            if close_h is not None and close_h > (open_h or 0) and leave > close_h:
                notes.append(f"Closes at {fmt_clock(close_h)}: shorten the visit or go earlier")
            if d["date"] and row is not None and is_closed_on(row.get("_weekly_off", ""), d["date"]):
                notes.append(f"Weekly off on {d['date'].strftime('%A')}: consider another day")
            slot = s.get("slot")
            s.update(arrive=arrive, leave=leave, travel_km=travel, notes=notes, lunch_before=lunch,
                     slot_ok=slot is None or SLOT_WINDOWS[slot][0] <= (arrive % 24) <= SLOT_WINDOWS[slot][1])
            new.append(s)
            t, prev = leave, s
        cen = (float(np.mean([x["lat"] for x in new])), float(np.mean([x["lon"] for x in new]))) if new else d["centroid"]
        out.append(dict(d, stops=new, km=km_total, end=t, centroid=cen, overloaded=(t - start_hour) > limit_h))
    return out


def render_plan_editor(plan, subset_df, start_hour, limit_h, city):
    """Lets the user move stops between days, change visit time, add a note or skip a stop."""
    rows = {r["_name"]: r for _, r in subset_df.iterrows()}
    nd = len(plan)
    names = sorted(s["name"] for d in plan for s in d["stops"])
    sig = hashlib.md5("|".join([city, str(nd), str(st.session_state.get("ed_ver", 0))] + names).encode()).hexdigest()[:8]
    seed = st.session_state.get("ed_seed", {})
    edits = {}
    with st.expander("✏️ Customise this plan: move, shorten or skip stops", expanded=False):
        st.caption("Change a stop's day, how long you stay, or skip it. Times and the route map update automatically.")
        for d in plan:
            for s in d["stops"]:
                k, sd = f"ed_{sig}_{slug(s['name'])}", seed.get(s["name"], {})
                defaults = {f"{k}_d": min(max(int(sd.get("day", d["day"])), 1), nd), f"{k}_h": float(sd.get("hours", s["hours"])),
                            f"{k}_n": sd.get("note", ""), f"{k}_s": bool(sd.get("skip", False))}
                for key, val in defaults.items():
                    st.session_state.setdefault(key, val)
                st.markdown(f"**{s['name']}** · planned for Day {d['day']}")
                c1, c2, c3 = st.columns(3)
                c1.selectbox("Day", list(range(1, nd + 1)), key=f"{k}_d")
                c2.number_input("Hours here", 0.5, 10.0, step=0.5, key=f"{k}_h")
                c3.checkbox("Skip this stop", key=f"{k}_s")
                st.text_input("Note (optional)", key=f"{k}_n", placeholder="e.g. book tickets online, bring ID")
                edits[s["name"]] = {"day": st.session_state[f"{k}_d"], "hours": st.session_state[f"{k}_h"],
                                    "skip": st.session_state[f"{k}_s"], "note": st.session_state[f"{k}_n"]}
                st.divider()
        if st.button("↩️ Reset all changes", key="reset_plan_edits_btn_v2"):
            st.session_state["ed_ver"] = st.session_state.get("ed_ver", 0) + 1
            st.session_state["ed_seed"] = {}
            st.rerun()
    orig = {s["name"]: (d["day"], s["hours"]) for d in plan for s in d["stops"]}
    changed = {n: e for n, e in edits.items() if e["skip"] or e["note"] or (e["day"], e["hours"]) != orig[n]}
    st.session_state["ed_current"] = changed
    if not changed:
        return plan
    st.caption(f"✏️ {len(changed)} stop(s) customised.")
    return apply_plan_edits(plan, edits, rows, start_hour, limit_h)


# ---------- Trip calendar ----------
def _cal_shift(delta):
    m = st.session_state["cal_month"] + delta
    y = st.session_state["cal_year"] + (m - 1) // 12
    st.session_state["cal_month"], st.session_state["cal_year"] = (m - 1) % 12 + 1, y


def _cal_today():
    t = datetime.date.today()
    st.session_state["cal_year"], st.session_state["cal_month"] = t.year, t.month


def _cal_apply_range():
    r = st.session_state.get("cal_range")
    if isinstance(r, (tuple, list)) and len(r) == 2:
        st.session_state["itin_start_date"], st.session_state["itin_return_date"] = r


def _use_dates(start, end):
    today = datetime.date.today()
    start = max(start, today)
    st.session_state["itin_start_date"], st.session_state["itin_return_date"] = start, max(end, start)


def _remove_special_day(i):
    prof = st.session_state["profile"]
    days = prof.get("custom_days", [])
    if 0 <= i < len(days):
        days.pop(i)
        profile_save()


def page_calendar():
    st.header("📅 Trip Calendar")
    st.caption("Holidays and festivals (they usually mean crowds and higher prices), long weekends, and your own special days. "
               "Festival dates that depend on the moon can shift by a day.")
    st.markdown(CALENDAR_CSS, unsafe_allow_html=True)
    today = datetime.date.today()
    st.session_state.setdefault("cal_year", today.year)
    st.session_state.setdefault("cal_month", today.month)
    s, e = st.session_state.get("itin_start_date"), st.session_state.get("itin_return_date")
    if not isinstance(s, datetime.date) or not isinstance(e, datetime.date):
        s, e = today + datetime.timedelta(days=1), today + datetime.timedelta(days=2)
        st.session_state["itin_start_date"], st.session_state["itin_return_date"] = s, e
    if st.session_state.get("_cal_sync") != (s, e):
        st.session_state["cal_range"], st.session_state["_cal_sync"] = (s, e), (s, e)
    st.date_input("Pick your trip dates (start and return)", min_value=today, max_value=today + datetime.timedelta(days=365 * 3),
                  format="DD/MM/YYYY", key="cal_range", on_change=_cal_apply_range,
                  help="These dates are shared with the Itinerary planner.")

    year, month = st.session_state["cal_year"], st.session_state["cal_month"]
    custom = st.session_state["profile"].get("custom_days", [])
    with st.container(key="cal_nav"):
        n1, n2, n3 = st.columns(3)
        n1.button("◀ Prev", key="cal_prev", on_click=_cal_shift, args=(-1,), use_container_width=True)
        n2.button("Today", key="cal_today", on_click=_cal_today, use_container_width=True)
        n3.button("Next ▶", key="cal_next", on_click=_cal_shift, args=(1,), use_container_width=True)
    st.markdown(f"### {MONTHS[month - 1]} {year}")
    st.markdown(month_calendar_html(year, month, holidays_for_date, s, e, today, custom), unsafe_allow_html=True)
    if year not in INDIA_HOLIDAYS:
        st.caption("Holiday data covers 2026 to 2028. Check the official calendar for other years.")

    events = month_events(year, month, holidays_for_date, custom)
    st.markdown("#### This month")
    if not events:
        st.caption("No major holidays listed this month.")
    for date, label, kind in events:
        st.markdown(f"{'⭐' if kind == 'custom' else '🎉'} **{date.strftime('%a, %d %b')}** · {label}")

    breaks = [b for b in long_weekends(year, month, holidays_for_date) if b["end"] >= today]
    if breaks:
        st.markdown("#### Long weekend ideas")
        for i, b in enumerate(breaks):
            with st.container(border=True):
                st.markdown(f"**{b['start'].strftime('%a %d %b')} to {b['end'].strftime('%a %d %b')}** · {b['days']} days around {b['why']}")
                st.caption(b["tip"] + " Hotels and trains fill up early on these dates.")
                st.button("Use these dates for my trip", key=f"cal_use_{i}", on_click=_use_dates, args=(b["start"], b["end"]))

    span = min((e - s).days + 1, 60)
    on_trip = [(d, h[0]) for d in (s + datetime.timedelta(days=i) for i in range(max(span, 0))) for h in holidays_for_date(d)]
    if on_trip:
        st.info("On your trip dates: " + ", ".join(f"{n} ({d.strftime('%d %b')})" for d, n in sorted(set(on_trip))) +
                ". Expect crowds, and book stays and transport early.")

    st.markdown("#### My special days")
    st.caption("Birthdays, anniversaries or leave days you want to see on the calendar.")
    with st.form("special_day_form", clear_on_submit=True):
        f1, f2 = st.columns([1, 2])
        d_val = f1.date_input("Date", value=today, format="DD/MM/YYYY")
        label = f2.text_input("What is it?", placeholder="e.g. Anniversary, leave approved")
        if st.form_submit_button("Add special day", type="primary") and label.strip():
            st.session_state["profile"].setdefault("custom_days", []).append({"date": d_val.isoformat(), "label": label.strip()[:40]})
            profile_save()
            st.rerun()
    for i, item in enumerate(custom):
        c1, c2 = st.columns([5, 1])
        c1.markdown(f"⭐ {item['date']} · {item['label']}")
        c2.button("Remove", key=f"cal_rm_{i}", on_click=_remove_special_day, args=(i,))


# ---------- Packing list ----------
PACK_EXTRAS = ["Kids", "Senior citizens", "Photography", "Pilgrimage"]


def page_packing():
    st.header("🎒 What to carry")
    prof = st.session_state["profile"]
    summ = st.session_state.get("trip_itinerary_summary")
    pool, src = df.iloc[0:0], ""
    if summ:
        names = [x["name"] for d in summ["days"] for x in d["stops"]]
        pool, src = df[df["_name"].isin(names) & (df["_city"] == summ["city"])], f"your {summ['city']} itinerary"
    if pool.empty and st.session_state["favorites"]:
        pool = df[df.apply(lambda r: f"{r['_name']}_{r['_city']}" in set(st.session_state["favorites"]), axis=1)]
        src = "your favourites"
    if pool.empty:
        cities = sorted(c for c in active_df["_city"].dropna().unique() if c)
        if not cities:
            st.info("No cities available in the current selection.")
            return
        keep("pack_city", cities[0], cities)
        city = st.selectbox("Which destination is this list for?", cities, key="pack_city")
        pool = active_df[active_df["_city"] == city].sort_values("_rating", ascending=False).head(8)
        src = f"popular places in {city}"
        st.caption("Tip: build an itinerary first and this list will match your exact stops.")
    s_date = st.session_state.get("itin_start_date")
    d_month = s_date.month if isinstance(s_date, datetime.date) else datetime.date.today().month
    d_days = len(summ["days"]) if summ else 3
    c1, c2, c3 = st.columns(3)
    keep("pack_month", MONTHS[d_month - 1], MONTHS)
    month = c1.selectbox("Travel month", MONTHS, key="pack_month")
    keep("pack_days", d_days)
    days = c2.number_input("Trip days", 1, 30, key="pack_days")
    keep("pack_transport", "Train", ["Train", "Bus", "Car", "Flight"])
    transport = c3.selectbox("Going by", ["Train", "Bus", "Car", "Flight"], key="pack_transport")
    keep("pack_extras", [], PACK_EXTRAS, multi=True)
    extras = st.multiselect("Travelling with / interests", PACK_EXTRAS, key="pack_extras")

    kinds = place_kinds(pool["_type"].tolist() + pool["_name"].tolist())
    _dslr = pool["_dslr"] if "_dslr" in pool.columns else [""] * len(pool)  # older cached data may lack this column
    no_dslr = [n for n, v in zip(pool["_name"], _dslr) if str(v).strip().lower().startswith("no")]
    plist = build_packing_list(MONTHS.index(month) + 1, days, kinds, transport, extras, pool["_state"].unique(), no_dslr)
    custom = st.session_state.setdefault("pack_custom", [])
    cats = list(plist.items())
    if custom:
        cats.append(("✍️ My own items", [(slug("custom-" + c), c) for c in custom]))
    saved = prof.setdefault("packing", {}).get("checked", {})
    ids = [iid for _, items in cats for iid, _l in items]
    for iid in ids:
        st.session_state.setdefault(f"pk_{iid}", bool(saved.get(iid, False)))
    done = sum(1 for iid in ids if st.session_state[f"pk_{iid}"])
    st.caption(f"Built from {src} · {month} · {len(kinds)} place type(s) detected")
    st.progress(done / max(len(ids), 1), text=f"{done} of {len(ids)} packed")

    for i in range(0, len(cats), 2):
        for col, (cat, items) in zip(st.columns(2), cats[i:i + 2]):
            with col, st.container(border=True):
                st.markdown(f"**{cat}**")
                for iid, label in items:
                    st.checkbox(label, key=f"pk_{iid}")

    with st.form("pack_add_form", clear_on_submit=True):
        a1, a2 = st.columns([4, 1])
        new_item = a1.text_input("Add your own item", placeholder="e.g. Board game, sketchbook", label_visibility="collapsed")
        if a2.form_submit_button("Add", use_container_width=True) and new_item.strip():
            custom.append(new_item.strip()[:60])
            prof["packing"]["custom"] = list(custom)
            profile_save()
            st.rerun()

    now_checked = {iid: True for iid in ids if st.session_state[f"pk_{iid}"]}
    keep_old = {k: v for k, v in saved.items() if k not in ids}
    if {**keep_old, **now_checked} != saved:
        prof["packing"]["checked"] = {**keep_old, **now_checked}
        prof["packing"]["custom"] = list(custom)
        profile_save()
    remaining = [lb for _, items in cats for iid, lb in items if not st.session_state[f"pk_{iid}"]]
    st.session_state["trip_packing_summary"] = {"packed": len(ids) - len(remaining), "total": len(ids), "remaining": remaining}

    def _reset():
        for iid in ids:
            st.session_state[f"pk_{iid}"] = False
    b1, b2 = st.columns(2)
    b1.button("↩️ Uncheck everything", key="pack_reset", on_click=_reset, use_container_width=True)
    text = f"Packing list · {month} · {days} day(s)\n" + "".join(
        f"\n{cat}\n" + "".join(f"  [{'x' if st.session_state[f'pk_{iid}'] else ' '}] {lb}\n" for iid, lb in items) for cat, items in cats)
    b2.download_button("⬇️ Download list", text, file_name="packing_list.txt", mime="text/plain", key="pack_dl", use_container_width=True)


# ---------- Saved trips & profile ----------
def _load_trip(i):
    t = st.session_state["profile"]["saved_trips"][i]
    city = t["city"]
    st.session_state["itin_city"], st.session_state["itin_travelers"] = city, int(t.get("travelers", 2))
    st.session_state[f"itin_stops_{city}"] = list(t.get("stops", []))
    start, end = datetime.date.fromisoformat(t["start"]), datetime.date.fromisoformat(t["end"])
    today = datetime.date.today()
    if start <= today:
        shift = (today + datetime.timedelta(days=1)) - start
        start, end = start + shift, end + shift
    st.session_state["itin_start_date"], st.session_state["itin_return_date"] = start, end
    st.session_state["ed_seed"] = t.get("edits", {})
    st.session_state["ed_ver"] = st.session_state.get("ed_ver", 0) + 1
    st.session_state["_trip_loaded"] = t["title"]


def _delete_trip(i):
    trips = st.session_state["profile"]["saved_trips"]
    if 0 <= i < len(trips):
        trips.pop(i)
        profile_save()


def page_saved_trips():
    st.header("🧳 Saved trips")
    trips = st.session_state["profile"].get("saved_trips", [])
    if st.session_state.get("_trip_loaded"):
        st.success(f"Loaded “{st.session_state.pop('_trip_loaded')}”. Open Plan My Trip → Itinerary to continue.")
    if not trips:
        st.info("No saved trips yet. Build an itinerary and press “Save this trip”.")
    for i, t in enumerate(trips):
        with st.container(border=True):
            st.markdown(f"**{t['title']}**")
            st.caption(f"{t['start']} to {t['end']} · {t.get('travelers', 2)} traveller(s) · {len(t.get('stops', []))} stops · saved {t.get('saved', '')}")
            st.write(", ".join(t.get("stops", [])))
            b1, b2 = st.columns(2)
            b1.button("Open this trip", key=f"trip_load_{i}", on_click=_load_trip, args=(i,), type="primary", use_container_width=True)
            b2.button("Delete", key=f"trip_del_{i}", on_click=_delete_trip, args=(i,), use_container_width=True)


def page_profile():
    st.header("👤 My profile")
    prof, user = st.session_state["profile"], st.session_state["user"]
    if user["guest"]:
        st.info("You are browsing as a guest. Log out and create an account to keep your trips, favourites and packing lists.")
    with st.form("profile_form"):
        name = st.text_input("Display name", value=prof.get("name", ""))
        avatar = st.radio("Avatar", AVATARS, index=AVATARS.index(prof.get("avatar", AVATARS[0])) if prof.get("avatar") in AVATARS else 0, horizontal=True)
        home = st.text_input("Home city (your starting point)", value=prof.get("home_city", ""), placeholder="e.g. Kolkata")
        styles = st.multiselect("Travel style", TRAVEL_STYLES, default=[x for x in prof.get("styles", []) if x in TRAVEL_STYLES])
        if st.form_submit_button("Save profile", type="primary"):
            prof.update(name=name.strip() or prof.get("name", "Traveller"), avatar=avatar, home_city=home.strip(), styles=styles)
            if home.strip():
                st.session_state["ai_start_city"] = home.strip()
            profile_save()
            st.success("Profile saved.")


# ---------- Page shell: state bar on every page + phone tab bar ----------
def _on_top_state():
    set_global_state(st.session_state["top_state_sel"])


def render_top_bar():
    prof = st.session_state["profile"]
    nav1, nav2, nav3, nav4, nav5, nav6 = st.columns([2.3, 1, 1.25, 1, 1, 1.5])
    with nav1:
        st.markdown("<div class='brand-mark'>🌍 <b>TravelAI</b><small> SMART TOURIST PLANNER</small></div>", unsafe_allow_html=True)
    with nav2:
        st.page_link(PAGES["home"], label="Home", icon="🏠")
    with nav3:
        st.page_link(PAGES["explore"], label="Destinations", icon="📍")
    with nav4:
        st.page_link(PAGES["ai"], label="For you", icon="🤖")
    with nav5:
        st.page_link(PAGES["plan"], label="Plan", icon="🗓️")
    with nav6:
        with st.popover(f"{prof.get('avatar', '🧭')} {prof.get('name', 'Traveller').split()[0]}  ▾", use_container_width=True):
            st.markdown(f"<div class='profile-pop'><div class='profile-avatar'>{html.escape(prof.get('avatar','🧭'))}</div><h3>{html.escape(prof.get('name','Traveller'))}</h3><p>{'Guest session' if st.session_state['user']['guest'] else '@' + html.escape(st.session_state['user']['username'])}</p></div>", unsafe_allow_html=True)
            if st.button("👤  My Profile", key="top_profile_btn", use_container_width=True):
                st.session_state["sec_mytrip"] = MYTRIP_SECTIONS[2]
                st.switch_page(PAGES["mytrip"])
            if st.button("⭐  My Travel", key="top_mytravel_btn", use_container_width=True):
                st.session_state["sec_mytrip"] = MYTRIP_SECTIONS[0]
                st.switch_page(PAGES["mytrip"])
            if st.button("Log out", key="top_logout_btn", use_container_width=True, on_click=_logout):
                st.rerun()
    st.markdown("<div class='top-nav-rule'></div>", unsafe_allow_html=True)


def render_mobile_nav():
    # Navigation is in the top bar on all screen sizes; no duplicate bottom bar.
    return None


def shell(fn):
    @functools.wraps(fn)
    def _page():
        render_top_bar()
        fn()
    return _page


def page_explore_hub():
    # Give the Explore / Place details switch its own clear navigation box.
    with st.container(border=True):
        section = _section_radio("sec_explore", EXPLORE_SECTIONS)
    (page_explore if section == EXPLORE_SECTIONS[0] else page_details)()


def page_plan_hub():
    section = _section_radio("sec_plan", PLAN_SECTIONS)
    dict(zip(PLAN_SECTIONS, [page_itinerary, page_packing, page_budget]))[section]()


def page_mytrip_hub():
    section = _section_radio("sec_mytrip", MYTRIP_SECTIONS)
    dict(zip(MYTRIP_SECTIONS, [page_favorites, page_saved_trips, page_profile, page_summary]))[section]()


PAGES = {
    "home": st.Page(shell(page_home), title="Home", icon="🏠", default=True),
    "ai": st.Page(shell(page_ai), title="AI Recommendations", icon="🤖", url_path="recommendations"),
    "explore": st.Page(shell(page_explore_hub), title="Explore", icon="📍", url_path="explorer"),
    "plan": st.Page(shell(page_plan_hub), title="Plan My Trip", icon="🗓️", url_path="plan"),
    "mytrip": st.Page(shell(page_mytrip_hub), title="My Travel", icon="⭐", url_path="mytrip"),
}
current_page = st.navigation([PAGES[k] for k in ("home", "ai", "explore", "plan", "mytrip")], position="hidden")

# Keep the budget inputs alive while the user visits other pages.
_PERSIST_KEYS = {
    "tb_group", "tb_days", "tb_tier", "tb_dist", "tb_mode", "tb_method",
    "ai_start_city", "ai_month", "ai_traveler", "ai_days", "ai_tier", "ai_reach", "ai_multi_category",
    "ai_query", "ai_top_n", "ex_search", "ex_rating", "ex_sort", "ex_unlisted", "ex_cards_n", "ex_layout",
    "near_pick", "near_radius", "near_types", "itin_city", "itin_days",
    "itin_travelers", "itin_date", "itin_start", "itin_limit", "detail_select", "trans_place_select",
    "trans_radius", "trans_travellers", "sec_explore", "sec_plan", "sec_mytrip",
    "itin_start_date", "itin_return_date", "cal_year", "cal_month", "pack_month", "pack_days", "pack_transport",
    "pack_extras", "pack_city", "pack_custom", "ed_seed", "ed_ver",
}
_PERSIST_PREFIXES = ("ex_types_", "ex_fee_", "itin_stops_", "pk_", "ed_")
for _k in list(st.session_state.keys()):
    if _k in _PERSIST_KEYS or (isinstance(_k, str) and _k.startswith(_PERSIST_PREFIXES)):
        st.session_state[_k] = st.session_state[_k]
# True only on the first run after arriving on a page.
st.session_state["_arrived"] = st.session_state.get("_prev_page") != current_page.title
st.session_state["_prev_page"] = current_page.title

current_page.run()

