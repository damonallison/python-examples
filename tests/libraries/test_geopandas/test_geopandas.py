"""
geopandas: geo extensions for pandas. Uses shapely for GIS functions.

CRS = coordinate reference system (projection)

Only one GeoSeries in a GeoDataFrame is considered the "active" geometry, which
means that all geometric operations applied to a GeoDataFrame happen on this
active geometry.

Shapely:

* Point: Point() / MultiPoint()
* Curve: LineString() / LinearRing() / MultiLineString()
* Surface: Polygon() / MultiPolygon()

"""
