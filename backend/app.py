# backend/app.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple
import os
import traceback
import re

from flask import Flask, request, render_template
from flask_cors import CORS

# --- Use certifi roots to avoid SSL issues in geopy/requests ---
try:
    import certifi  # type: ignore
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

import networkx as nx
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

from graph_loader import load_graph, nearest_node
from routing import (
    astar_shortest_path,
    route_total_length_m,
    route_coordinates,
    render_folium_map,
)

# ---------- Flask ----------
app = Flask(__name__, template_folder="../templates", static_folder="../static")
CORS(app)

# ---------- Paths ----------
ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT_DIR / "static"
MAP_HTML = STATIC_DIR / "route_map.html"

# ---------- Load road graph ----------
G: nx.MultiDiGraph = load_graph()

# ---------- Geocoder ----------
geolocator = Nominatim(user_agent="kl-shortest-route-demo")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.0, swallow_exceptions=True)

# ---------- Flexible geocoding ----------
_KL_VIEWBOX = {  # rough bbox for Klang Valley
    "min_lon": 101.3,
    "min_lat": 2.7,
    "max_lon": 101.95,
    "max_lat": 3.35,
}

ALIASES = {
    "UPM": "Universiti Putra Malaysia",
    "KLCC": "Suria KLCC, Kuala Lumpur",
    "KLIA": "Kuala Lumpur International Airport",
    "PAVILION": "Pavilion Kuala Lumpur",
    "MID VALLEY": "Mid Valley Megamall, Kuala Lumpur",
    "TIMES SQUARE": "Berjaya Times Square, Kuala Lumpur",
}

coord_pat = re.compile(r"^\s*(-?\d+(\.\d+)?)\s*,\s*(-?\d+(\.\d+)?)\s*$")


def _try_geocode(q: str, *, country_bias=True, viewbox_bias=True):
    # geopy expects viewbox=((south, west), (north, east)) -> ((lat, lon), (lat, lon))
    if country_bias and viewbox_bias:
        loc = geocode(
            q,
            exactly_one=True,
            language="en",
            country_codes="my",
            viewbox=(
                (_KL_VIEWBOX["min_lat"], _KL_VIEWBOX["min_lon"]),  # south, west
                (_KL_VIEWBOX["max_lat"], _KL_VIEWBOX["max_lon"]),  # north, east
            ),
            bounded=True,
            addressdetails=False,
        )
        if loc:
            return loc

    if country_bias:
        loc = geocode(
            q, exactly_one=True, language="en",
            country_codes="my", addressdetails=False
        )
        if loc:
            return loc

    return geocode(q, exactly_one=True, language="en", addressdetails=False)


def resolve_place_to_latlon(q: str) -> Tuple[float, float]:
    """
    Resolve user text to (lat, lon). Supports:
      - aliases (UPM/KLCC/KLIA, etc.)
      - raw 'lat,lon'
      - Malaysia/KL-biased search with global fallback
    """
    s = q.strip()
    if not s:
        raise ValueError("Empty location.")

    # Aliases
    up = s.upper()
    if up in ALIASES:
        s = ALIASES[up]

    # Raw coordinates "lat,lon"
    m = coord_pat.match(s)
    if m:
        return float(m.group(1)), float(m.group(3))

    # Add locality hint if missing
    s_aug = s if ("Malaysia" in s or "Kuala Lumpur" in s) else f"{s}, Kuala Lumpur, Malaysia"

    loc = (
        _try_geocode(s_aug, country_bias=True, viewbox_bias=True)
        or _try_geocode(s_aug, country_bias=True, viewbox_bias=False)
        or _try_geocode(s, country_bias=False, viewbox_bias=False)
    )
    if not loc:
        raise ValueError(f"Could not geocode: {q!r}. Try adding 'Kuala Lumpur' or use 'lat,lon'.")
    return float(loc.latitude), float(loc.longitude)


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/route")
def route_endpoint():
    try:
        data = request.form if request.form else request.json
        if not data:
            return render_template("index.html", error="Please provide both locations."), 400

        point_a = (data.get("pointA") or "").strip()
        point_b = (data.get("pointB") or "").strip()
        if not point_a or not point_b:
            return render_template("index.html", error="Both Point A and Point B are required."), 400

        # Geocode
        a_lat, a_lon = resolve_place_to_latlon(point_a)
        b_lat, b_lon = resolve_place_to_latlon(point_b)

        # Snap & route
        src = nearest_node(G, a_lat, a_lon)
        dst = nearest_node(G, b_lat, b_lon)
        node_route = astar_shortest_path(G, src, dst)

        # Distance + map
        dist_m = route_total_length_m(G, node_route)
        coords = route_coordinates(G, node_route)
        render_folium_map(coords, point_a, point_b, str(MAP_HTML))

        return render_template(
            "result.html",
            distance_km=f"{dist_m/1000:.2f}",
            map_rel_path="route_map.html",
            point_a=point_a,
            point_b=point_b
        )
    except ValueError as ve:
        return render_template("index.html", error=str(ve)), 400
    except Exception as e:
        traceback.print_exc()
        return render_template("index.html", error=f"Unexpected error: {e}"), 500


if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    app.config["JSON_AS_ASCII"] = False
    app.run(host="127.0.0.1", port=5000, debug=True)
