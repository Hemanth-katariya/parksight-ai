"""ParkSight analytics engine."""

# H3 API compatibility layer (v3 vs v4)
try:
    import h3
    if hasattr(h3, 'geo_to_h3'):
        # v3 API
        h3_latlng_to_cell = h3.geo_to_h3
        h3_cell_to_boundary = lambda idx, geo_json=False: h3.h3_to_geo_boundary(idx, geo_json=geo_json)
    elif hasattr(h3, 'latlng_to_cell'):
        # v4 API
        h3_latlng_to_cell = h3.latlng_to_cell
        h3_cell_to_boundary = lambda idx, geo_json=False: h3.cell_to_boundary(idx)
    else:
        h3_latlng_to_cell = h3.geo_to_h3
        h3_cell_to_boundary = lambda idx, geo_json=False: h3.h3_to_geo_boundary(idx, geo_json=geo_json)
except ImportError:
    pass
