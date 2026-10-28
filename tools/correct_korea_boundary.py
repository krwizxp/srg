import argparse, hashlib, io, json, math, pathlib, struct, zipfile
import numpy as np
import pyproj, shapefile, shapely
from shapely.geometry import LineString, Point, Polygon, MultiPolygon, box, shape, mapping
from shapely.ops import split, substring
SOURCE_DATE = '20250630'
PATCH_BOUNDS = (920000, 1970000, 1085000, 2080000)

def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def load_reference(path):
    with zipfile.ZipFile(path) as z:
        b = next((n[:-4] for n in z.namelist() if n.endswith('.shp')))
        crs = pyproj.CRS.from_wkt(z.read(b + '.prj').decode())
        t = pyproj.Transformer.from_crs(crs, 5179, always_xy=True)
        r = shapefile.Reader(shp=io.BytesIO(z.read(b + '.shp')), shx=io.BytesIO(z.read(b + '.shx')), dbf=io.BytesIO(z.read(b + '.dbf')), encoding='cp949')
        return shapely.union_all([shapely.transform(shape(s.__geo_interface__), t.transform, interleaved=False) for s in r.shapes()])

def polygons(g):
    if g.is_empty:
        return []
    if g.geom_type == 'Polygon':
        return [g]
    if g.geom_type == 'MultiPolygon':
        return list(g.geoms)
    raise ValueError(f'Expected polygonal geometry, got {g.geom_type}')

def split_dmz(dmz, mdl):
    c = list(mdl.coords)

    def extend(a, b):
        dx, dy = (a[0] - b[0], a[1] - b[1])
        length = math.hypot(dx, dy)
        return (a[0] + 12000 * dx / length, a[1] + 12000 * dy / length)
    divider = LineString([extend(c[0], c[1])] + c + [extend(c[-1], c[-2])])
    pieces = sorted(split(dmz, divider).geoms, key=lambda p: p.centroid.y)
    if len(pieces) != 2:
        raise ValueError('Changed reference topology; side identification needs reassessment')
    return (pieces[0], pieces[1])

def rebuild_mainland(main, mdl):
    if mdl.coords[0][0] > mdl.coords[-1][0]:
        mdl = shapely.reverse(mdl)
    ring = LineString(main.exterior.coords)
    crosses = ring.intersection(mdl)
    candidates = [p for p in crosses.geoms if p.geom_type == 'Point'] if hasattr(crosses, 'geoms') else [crosses]
    if not candidates or any((p.geom_type != 'Point' for p in candidates)):
        raise ValueError('Expected discrete original/reference crossings')
    ordered = sorted([(mdl.project(p), p) for p in candidates], key=lambda x: x[0])
    (start, a), (end, b) = (ordered[0], ordered[-1])
    line = substring(mdl, start, end)
    cs = np.asarray(ring.coords)
    dist = np.r_[0, np.cumsum(np.hypot(*np.diff(cs, axis=0).T))]
    length = dist[-1]
    pa, pb = (ring.project(a), ring.project(b))

    def ring_arc(lo, span):
        hi = lo + span
        first = np.array(ring.interpolate(float(lo % length)).coords[0])
        last = np.array(ring.interpolate(float(hi % length)).coords[0])
        if hi <= length:
            return np.vstack([first, cs[(dist > lo) & (dist < hi)], last])
        return np.vstack([first, cs[(dist > lo) & (dist < length)], cs[(dist >= 0) & (dist < hi - length)], last])
    ab = (pb - pa) % length
    ba = (pa - pb) % length
    if min(ab, ba) > 500000:
        raise ValueError('Selected arc is too long for the northern frontier')
    if ab < ba:
        old_arc = ring_arc(pa, ab)
        retained = ring_arc(pb, ba)
        coords = np.vstack([np.asarray(line.coords), retained[1:]])
    else:
        old_arc = ring_arc(pb, ba)
        retained = ring_arc(pa, ab)
        coords = np.vstack([np.asarray(line.coords)[::-1], retained[1:]])
    rebuilt = Polygon(coords, [list(r.coords) for r in main.interiors])
    if not rebuilt.is_valid:
        raise ValueError(shapely.is_valid_reason(rebuilt))
    return (rebuilt, line, LineString(old_arc), {'start_chainage_m': start, 'end_chainage_m': end, 'full_reference_length_m': mdl.length, 'western_reference_not_replaced_m': start, 'eastern_reference_not_replaced_m': mdl.length - end, 'intersection_count': len(ordered), 'synthetic_frontier_connectors': 0})

def write_shapefile(base, g, wkt):
    g = shapely.orient_polygons(g, exterior_cw=True)
    rings = []
    for p in polygons(g):
        rings.extend([np.asarray(p.exterior.coords), *[np.asarray(r.coords) for r in p.interiors]])
    counts = np.array([len(r) for r in rings], dtype=np.int64)
    npoints = int(counts.sum())
    nparts = len(rings)
    offsets = np.r_[0, np.cumsum(counts)[:-1]].astype('<i4')
    bounds = g.bounds
    content_bytes = 44 + 4 * nparts + 16 * npoints

    def header(words):
        return struct.pack('>7i', 9994, 0, 0, 0, 0, 0, words) + struct.pack('<2i8d', 1000, 5, *bounds, 0, 0, 0, 0)
    with open(str(base) + '.shp', 'wb') as f:
        f.write(header((108 + content_bytes) // 2))
        f.write(struct.pack('>2i', 1, content_bytes // 2))
        f.write(struct.pack('<i4d2i', 5, *bounds, nparts, npoints))
        f.write(offsets.tobytes())
        for r in rings:
            f.write(np.asarray(r, dtype='<f8').tobytes())
    with open(str(base) + '.shx', 'wb') as f:
        f.write(header(54))
        f.write(struct.pack('>2i', 50, content_bytes // 2))
    with shapefile.Writer(dbf=str(base) + '.dbf') as w:
        w.field('MASK_TYPE', 'C', 20)
        w.field('SGIS_DATE', 'C', 8)
        w.field('MDL_YEAR', 'C', 4)
        w.field('BUILD_DATE', 'C', 8)
        w.record('REFERENCE_CORRECTED', SOURCE_DATE, '2024', '20261028')
    pathlib.Path(str(base) + '.prj').write_text(wkt)
    pathlib.Path(str(base) + '.cpg').write_text('UTF-8')
    return {'polygon_components': len(polygons(g)), 'rings': nparts, 'vertices': npoints}

def export_geojson(path, features):
    t = pyproj.Transformer.from_crs(5179, 4326, always_xy=True)
    converted = [{'type': 'Feature', 'properties': props, 'geometry': mapping(shapely.transform(g, t.transform, interleaved=False))} for props, g in features if not g.is_empty]
    path.write_text(json.dumps({'type': 'FeatureCollection', 'features': converted}, ensure_ascii=False, separators=(',', ':')) + '\n')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sgis-dir', required=True, type=pathlib.Path)
    ap.add_argument('--kei-mdl', required=True, type=pathlib.Path)
    ap.add_argument('--kei-dmz', required=True, type=pathlib.Path)
    ap.add_argument('--out-dir', required=True, type=pathlib.Path)
    ap.add_argument('--union-cache', type=pathlib.Path, help='Optional local cache of the exact SGIS union; hash is recorded')
    args = ap.parse_args()
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    files = list(args.sgis_dir.rglob('bnd_sido_00_2025_2Q.shp'))
    if len(files) != 1:
        raise ValueError('Expected one SGIS 2025 Q2 sido layer')
    src = files[0]
    wkt = src.with_suffix('.prj').read_text()
    if pyproj.CRS.from_wkt(wkt).to_epsg() != 5179:
        raise ValueError('Unexpected SGIS CRS')
    reader = shapefile.Reader(str(src))
    records = [r.as_dict() for r in reader.records()]
    if len(records) != 17 or any((r['BASE_DATE'] != SOURCE_DATE for r in records)):
        raise ValueError('Unexpected SGIS version')
    sg = shapely.from_wkb(args.union_cache.read_bytes()) if args.union_cache else shapely.union_all([shape(s.__geo_interface__) for s in reader.shapes()])
    if not sg.is_valid or sg.geom_type != 'MultiPolygon':
        raise ValueError('Invalid source union')
    mdl = load_reference(args.kei_mdl)
    dmz = load_reference(args.kei_dmz)
    if mdl.geom_type != 'LineString' or not mdl.is_simple or (not dmz.is_valid):
        raise ValueError('Invalid reference geometries')
    south, north = split_dmz(dmz, mdl)
    mainland = max(sg.geoms, key=lambda g: g.area)
    rebuilt, used_line, old_arc, joins = rebuild_mainland(mainland, mdl)
    window = box(*PATCH_BOUNDS)
    near = []
    far = []
    for g in sg.geoms:
        if g == mainland:
            continue
        (near if g.intersects(window) else far).append(g)
    merged = shapely.union_all([rebuilt.difference(north)] + [g.difference(north) for g in near])
    corrected = MultiPolygon(polygons(merged) + far)
    if not corrected.is_valid:
        raise ValueError(shapely.is_valid_reason(corrected))
    old_local = sg.intersection(window)
    new_local = merged.intersection(window)
    added = new_local.difference(old_local)
    removed = old_local.difference(new_local)
    remaining = south.difference(merged)
    before_south = south.difference(old_local)
    overlap_before = old_local.intersection(north)
    overlap_after = new_local.intersection(north)
    old_main_local = mainland.intersection(window)
    new_main_local = rebuilt.intersection(window)
    delta = shapely.symmetric_difference(old_main_local, new_main_local).area
    area_identity = abs(rebuilt.area - mainland.area - (new_main_local.area - old_main_local.area))
    if area_identity > 0.01:
        raise ValueError('Mainland area changed outside the declared patch')
    if not window.covers(dmz):
        raise ValueError('Reference DMZ extends outside the declared patch')

    def outside_vertices(ring):
        v = np.asarray(ring.coords)[:-1]
        x0, y0, x1, y1 = PATCH_BOUNDS
        v = v[(v[:, 0] < x0) | (v[:, 0] > x1) | (v[:, 1] < y0) | (v[:, 1] > y1)]
        return v[np.lexsort((v[:, 1], v[:, 0]))]
    conserved_vertices = outside_vertices(mainland.exterior)
    if not np.array_equal(conserved_vertices, outside_vertices(rebuilt.exterior)):
        raise ValueError('Original exterior vertices outside the patch changed')
    if not all((shapely.equals_exact(x, y, 0) for x, y in zip(mainland.interiors, rebuilt.interiors))):
        raise ValueError('Original mainland holes changed')
    if overlap_after.area > 1e-06 or added.intersection(north).area > 1e-06:
        raise ValueError('Reference north overlap remains')
    coastal_review = box(1074000, 2064000, 1077000, 2070000)
    remaining_inland = remaining.difference(coastal_review)
    if remaining_inland.area > 0.0001:
        raise ValueError('An inland southern-reference omission remains')
    t = pyproj.Transformer.from_crs(4326, 5179, always_xy=True)
    shapely.prepare(corrected)
    shapely.prepare(sg)
    probes = {'Seoul': (126.978, 37.5665, True), 'Dokdo_east': (131.8695556, 37.2407778, True), 'Dokdo_west': (131.865, 37.2418333, True), 'Marado': (126.267, 33.118, True), 'Baengnyeong': (124.667, 37.95, True), 'Daecheong': (124.705, 37.825, True), 'Yeonpyeong': (125.704, 37.665, True), 'Ganghwa': (126.488, 37.746, True), 'Goseong': (128.369, 38.585, True), 'Pyongyang': (125.754, 39.039, False), 'Kaesong': (126.554, 37.971, False), 'Tsushima': (129.325, 34.205, False), 'West_sea': (125.6, 35.5, False), 'Restored_inland': (127.98328481219534, 38.302453273850496, True), 'Removed_reference_north': (126.88366747448389, 38.109516351129066, False)}
    probe_results = {}
    for name, (lon, lat, expected) in probes.items():
        p = Point(t.transform(lon, lat))
        actual = bool(corrected.covers(p))
        prior = bool(sg.covers(p))
        if actual != expected:
            raise ValueError(f'Probe failed: {name}')
        probe_results[name] = {'lon': lon, 'lat': lat, 'before_inside': prior, 'after_inside': actual, 'expected_inside': expected}
    border = merged.boundary.intersection(dmz.buffer(5000))
    pts = shapely.line_interpolate_point(used_line, np.arange(0, used_line.length, 100))
    dist = shapely.distance(pts, border)
    if max(dist) > 0.001:
        raise ValueError(f'Rebuilt frontier does not follow reference: {max(dist)} m')
    base = out / 'land_mask_5179'
    shape_stats = write_shapefile(base, corrected, wkt)
    check_reader = shapefile.Reader(str(base) + '.shp')
    decoded = shape(check_reader.shape(0).__geo_interface__)
    if not decoded.is_valid or abs(decoded.area - corrected.area) > 0.01 or (not shapely.equals_exact(shapely.normalize(decoded), shapely.normalize(corrected), 0)):
        raise ValueError('SHP round-trip failed')
    export_geojson(out / 'changes_wgs84.geojson', [({'change': 'RESTORED_SOUTH'}, added), ({'change': 'REMOVED_REFERENCE_NORTH'}, removed)])
    export_geojson(out / 'review_wgs84.geojson', [({'status': 'OUTSIDE_ADOPTED_SGIS_COAST', 'decision': 'NOT_ADDED', 'note': 'Not certified water/land or operational boundary'}, remaining)])
    export_geojson(out / 'frontier_wgs84.geojson', [({'kind': 'APPLIED_KEI_MDL', 'source_year': '2024'}, used_line)])
    input_paths = [src.with_suffix('.' + e) for e in ['shp', 'shx', 'dbf', 'prj', 'cpg']] + [args.kei_mdl, args.kei_dmz]
    areas = {'source_total_km2': sg.area / 1000000.0, 'corrected_total_km2': corrected.area / 1000000.0, 'restored_south_km2': added.area / 1000000.0, 'removed_reference_north_km2': removed.area / 1000000.0, 'before_overlap_north_dmz_km2': overlap_before.area / 1000000.0, 'after_overlap_north_dmz_km2': overlap_after.area / 1000000.0, 'before_missing_south_dmz_km2': before_south.area / 1000000.0, 'after_missing_south_dmz_km2': remaining.area / 1000000.0, 'after_missing_south_dmz_outside_coast_review_km2': remaining_inland.area / 1000000.0}
    result = {'build_date': '2026-10-28', 'status': 'PUBLIC_REFERENCE_CORRECTED_NOT_OPERATIONALLY_CERTIFIED', 'crs': 'EPSG:5179', 'patch_bounds_epsg5179': PATCH_BOUNDS, 'areas': areas, 'joins': joins, 'regression_probes': probe_results, 'checks': {'valid_source_union': True, 'valid_corrected_polygon': True, 'no_reference_north_overlap': True, 'no_unresolved_southern_reference_omission_outside_coastal_review': True, 'new_frontier_sampled_max_distance_m': float(max(dist)), 'sample_spacing_m': 100, 'outside_patch_mainland_area_identity_error_m2': area_identity, 'unchanged_distant_polygon_components': len(far), 'unchanged_mainland_exterior_vertices_outside_patch': int(len(conserved_vertices)), 'original_mainland_holes_retained': True, 'independent_shapefile_roundtrip': True}, 'geometry': shape_stats, 'inputs': [{'file': p.name, 'bytes': p.stat().st_size, 'sha256': digest(p)} for p in input_paths], 'software': {'shapely': shapely.__version__, 'GEOS': shapely.geos_version_string, 'pyproj': pyproj.__version__, 'PROJ': pyproj.proj_version_str, 'pyshp': shapefile.__version__, 'numpy': np.__version__}, 'sources': {'SGIS': 'https://www.data.go.kr/data/15129688/fileData.do', 'MDL': 'https://data.neins.go.kr/detail/dts-eh03eVuTQu', 'DMZ': 'https://data.neins.go.kr/detail/dts-FeJLArLgg1'}, 'policies': {'mainland': 'Replace the short northern exterior arc between first and last exact original/MDL crossings. Preserve original holes.', 'north': 'Subtract reference-north DMZ from all nearby components, including detached river polygons.', 'coast': 'No invented connector, seaward expansion, NLL extrapolation, or filling coastal review area.', 'islands': 'All 13690 distant original polygon components unchanged. Polygon component count is not an island count.', 'water': 'Original SGIS inland water policy retained; no claim that every included coordinate is dry land or publicly accessible.', 'precision': 'No simplification or coordinate rounding; EPSG:5179 projected areas, not official/ellipsoidal areas.', 'reference_extent': 'KEI metadata 2024; latest survey date is unconfirmed. End stretches not replaced are explicitly measured.', 'review': 'Remaining 0.371 km2 mismatch is outside the adopted SGIS coast, not proven missing terrestrial territory.', 'not_used': 'Uploaded N3A_G0010000.zip is not used to restore omitted northern regions.'}}
    if args.union_cache:
        result['local_union_cache_sha256'] = digest(args.union_cache)
    (out / 'validation.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'areas': areas, 'checks': result['checks'], 'geometry': shape_stats, 'joins': joins}, ensure_ascii=False, indent=2), flush=True)
if __name__ == '__main__':
    main()
