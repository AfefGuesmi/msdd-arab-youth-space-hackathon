import 'dart:math';

import 'package:flutter/foundation.dart' show compute;
import 'package:flutter/services.dart' show AssetManifest, rootBundle;
import 'package:latlong2/latlong.dart';

/// Reads the detections report (assets/files/mssd_debris_report.csv) and
/// turns it into the detected zones shown on the map.
///
/// The file has one row per detected object (about 32,000 rows):
///
///   Debris Type, Danger Level, Geolocation, Recommended Action,
///   Report Date, Source_File
///
/// Two things have to happen before those rows can be drawn:
///
/// 1. Geolocation is "northing, easting" in metres (UTM), not latitude /
///    longitude. The UTM zone and hemisphere come from the satellite tile
///    code inside Source_File (e.g. "..._16PCC_..." = zone 16, northern
///    hemisphere), and the position is converted to latitude / longitude.
/// 2. 32,000 separate dots would be unreadable, so detections of the same
///    debris type that fall in the same ~11 km cell are merged into ONE
///    zone: its centre is the average position and its polygon is the
///    outline around all of its detections.
///
/// Each zone is the same `Map<String, dynamic>` the rest of the app uses:
/// ocean, polygon (List<LatLng>), center (LatLng), type, danger, solution,
/// image (asset path).
/// A problem with the zones file that comes with a message fit to show.
class CsvZonesException implements Exception {
  final String message;
  const CsvZonesException(this.message);

  @override
  String toString() => message;
}

class CsvZonesLoader {
  CsvZonesLoader._();

  static const String assetPath = 'assets/files/mssd_debris_report.csv';

  // The three qualitative-result images, handed out in turn to the zones:
  // zone 1 gets r1, zone 2 gets r2, zone 3 gets r3, zone 4 starts over...
  static const List<String> imageAssets = [
    'assets/qualitative_results/r1.png',
    'assets/qualitative_results/r2.png',
    'assets/qualitative_results/r3.png',
  ];

  // Size of the merge cell, in degrees (0.1 degree is roughly 11 km).
  static const double _cellDegrees = 0.1;

  // Each detection is padded by this much (roughly 400 m) before the zone
  // outline is drawn, so a zone with a handful of detections is still a
  // visible shape instead of a line.
  static const double _padDegrees = 0.004;

  // Sea / region name for each satellite tile that appears in the report,
  // worked out from the converted coordinates. A tile that isn't listed
  // falls back to "Unknown area".
  static const Map<String, String> _regionByTile = {
    '16PCC': 'Gulf of Honduras',
    '16PDC': 'Gulf of Honduras',
    '16PEC': 'Gulf of Honduras',
    '16QED': 'Bay Islands, Caribbean Sea',
    '18QYF': 'Gulf of Gonave, Haiti',
    '48PZC': 'Da Nang coast, South China Sea',
    '51PTS': 'Manila Bay',
    '50LLR': 'Bali Sea',
  };

  /// Loads the CSV and builds the zones. The heavy work (reading 32,000
  /// rows) runs in a background isolate so the map never freezes.
  static Future<List<Map<String, dynamic>>> load() async {
    final String path = await _findCsvAsset();
    final String text = await rootBundle.loadString(path);
    try {
      return await compute(parseZones, text);
    } catch (_) {
      // If the background isolate can't be used for any reason, do the
      // same work right here instead of failing.
      return parseZones(text);
    }
  }

  // Finds the CSV inside assets/files/. The expected name is tried first,
  // but any .csv in that folder is accepted, so a different file name
  // doesn't break loading. If nothing is found, the error says exactly
  // what to fix.
  static Future<String> _findCsvAsset() async {
    try {
      final AssetManifest manifest =
          await AssetManifest.loadFromAssetBundle(rootBundle);
      final List<String> csvs = manifest
          .listAssets()
          .where((a) =>
              a.startsWith('assets/files/') && a.toLowerCase().endsWith('.csv'))
          .toList();
      if (csvs.contains(assetPath)) return assetPath;
      if (csvs.isNotEmpty) return csvs.first;
    } catch (_) {
      // Manifest not readable — fall back to the expected path below.
      return assetPath;
    }
    throw const CsvZonesException(
      'No .csv file found in assets/files/. Check that the file is in that '
      'folder, that pubspec.yaml lists "- assets/files/" under '
      'flutter: assets:, then run "flutter pub get" and fully restart the app '
      '(hot reload does not pick up new assets).',
    );
  }

  // Must stay a top-level-style static so `compute` can run it elsewhere.
  //
  // Speed notes: the file is read line by line and each row is folded
  // straight into its zone (no list of 32,000 rows is kept), cells are cut
  // with indexOf instead of one character at a time, the tile code of each
  // Source_File is looked up once and remembered, and the UTM constants are
  // computed once instead of once per row.
  static List<Map<String, dynamic>> parseZones(String csvText) {
    String text = csvText;
    if (text.isNotEmpty && text.codeUnitAt(0) == 0xFEFF) {
      text = text.substring(1);
    }
    final List<String> lines = text.split('\n');
    if (lines.length < 2) return [];

    final List<String> header = _splitLine(lines.first);
    final Map<String, int> col = {};
    for (int i = 0; i < header.length; i++) {
      col[header[i].trim().toLowerCase()] = i;
    }
    final int? iType = col['debris type'];
    final int? iDanger = col['danger level'];
    final int? iGeo = col['geolocation'];
    final int? iAction = col['recommended action'];
    final int? iSource = col['source_file'];
    if (iType == null || iGeo == null || iSource == null) return [];

    // The tile code, e.g. "16PCC" -> zone 16, band P.
    final RegExp tileRe = RegExp(r'_(\d{1,2})([C-X])([A-Z]{2})_');
    // Source_File -> its tile (there are far fewer files than rows).
    final Map<String, _Tile?> tiles = {};

    final Map<String, _Cluster> clusters = {};

    for (int li = 1; li < lines.length; li++) {
      final String line = lines[li];
      if (line.trim().isEmpty) continue;
      final List<String> row = _splitLine(line);
      if (row.length <= iGeo || row.length <= iType || row.length <= iSource) {
        continue;
      }
      final String type = row[iType].trim();
      if (type.isEmpty) continue;

      final String source = row[iSource];
      final _Tile? tile = tiles.putIfAbsent(source, () {
        final RegExpMatch? m = tileRe.firstMatch(source);
        if (m == null) return null;
        return _Tile(
          int.parse(m.group(1)!),
          m.group(2)!.compareTo('N') >= 0,
          '${m.group(1)}${m.group(2)}${m.group(3)}',
        );
      });
      if (tile == null) continue;

      final String geo = row[iGeo];
      final int comma = geo.indexOf(',');
      if (comma < 0) continue;
      final double? northing = double.tryParse(geo.substring(0, comma).trim());
      final double? easting = double.tryParse(geo.substring(comma + 1).trim());
      if (northing == null || easting == null) continue;

      final List<double> ll =
          _utmToLatLng(tile.zone, tile.north, easting, northing);
      final double lat = ll[0];
      final double lng = ll[1];

      final String key =
          '$type|${(lat / _cellDegrees).floor()}|${(lng / _cellDegrees).floor()}';
      final _Cluster c =
          clusters.putIfAbsent(key, () => _Cluster(type, tile.id));
      c.add(
        lat,
        lng,
        danger: iDanger != null && iDanger < row.length ? row[iDanger] : '',
        action: iAction != null && iAction < row.length ? row[iAction] : '',
      );
    }

    // One zone per cluster. Most severe first, then the largest.
    final List<_Cluster> ordered = clusters.values.toList()
      ..sort((a, b) {
        final int byDanger = _dangerRank(b.danger).compareTo(_dangerRank(a.danger));
        if (byDanger != 0) return byDanger;
        return b.count.compareTo(a.count);
      });

    final List<Map<String, dynamic>> zones = [];
    for (int i = 0; i < ordered.length; i++) {
      final _Cluster c = ordered[i];
      zones.add({
        'ocean': _regionByTile[c.tileId] ?? 'Unknown area',
        'polygon': c.outline(),
        'center': c.center,
        'type': c.type,
        'danger': c.danger,
        'solution':
            c.action.isEmpty ? 'No recommendation available' : c.action,
        'image': imageAssets[i % imageAssets.length],
      });
    }
    return zones;
  }

  // Cuts one CSV line into its cells. Understands quoted cells ("a, b")
  // and doubled quotes inside them. (Cells containing line breaks are not
  // supported — this report has none.)
  static List<String> _splitLine(String rawLine) {
    String line = rawLine;
    if (line.endsWith('\r')) line = line.substring(0, line.length - 1);

    final List<String> cells = [];
    final int len = line.length;
    int i = 0;
    while (i <= len) {
      if (i < len && line.codeUnitAt(i) == 0x22) {
        // quoted cell
        final StringBuffer buf = StringBuffer();
        int j = i + 1;
        while (true) {
          final int q = line.indexOf('"', j);
          if (q < 0) {
            buf.write(line.substring(j));
            j = len;
            break;
          }
          buf.write(line.substring(j, q));
          if (q + 1 < len && line.codeUnitAt(q + 1) == 0x22) {
            buf.write('"');
            j = q + 2;
          } else {
            j = q + 1;
            break;
          }
        }
        cells.add(buf.toString());
        // skip up to and including the comma after the closing quote
        final int next = line.indexOf(',', j);
        if (next < 0) break;
        i = next + 1;
        if (i == len) {
          cells.add('');
          break;
        }
      } else {
        final int next = line.indexOf(',', i);
        if (next < 0) {
          cells.add(line.substring(i));
          break;
        }
        cells.add(line.substring(i, next));
        i = next + 1;
        if (i == len) {
          cells.add('');
          break;
        }
      }
    }
    return cells;
  }

  static int _dangerRank(String danger) {
    switch (danger) {
      case 'Critical':
        return 4;
      case 'High':
        return 3;
      case 'Medium':
        return 2;
      case 'Low':
        return 1;
      default:
        return 0;
    }
  }

  // ---------------------------------------------------------------
  // UTM (WGS84) -> latitude / longitude
  // ---------------------------------------------------------------

  // dart:math has no sinh / cosh, so they are built from exp.
  static double _sinh(double x) => (exp(x) - exp(-x)) / 2;
  static double _cosh(double x) => (exp(x) + exp(-x)) / 2;

  // WGS84 / UTM constants, computed once.
  static const double _k0 = 0.9996;
  static const double _f = 1 / 298.257223563;
  static const double _n = _f / (2 - _f);
  static final double _bigA = 6378137.0 /
      (1 + _n) *
      (1 + _n * _n / 4 + _n * _n * _n * _n / 64);
  static final List<double> _beta = [
    _n / 2 - 2 * _n * _n / 3 + 37 * _n * _n * _n / 96 - _n * _n * _n * _n / 360,
    _n * _n / 48 + _n * _n * _n / 15 - 437 * _n * _n * _n * _n / 1440,
    17 * _n * _n * _n / 480 - 37 * _n * _n * _n * _n / 840,
    4397 * _n * _n * _n * _n / 161280,
  ];
  static final List<double> _delta = [
    2 * _n - 2 * _n * _n / 3 - 2 * _n * _n * _n + 116 * _n * _n * _n * _n / 45,
    7 * _n * _n / 3 - 8 * _n * _n * _n / 5 - 227 * _n * _n * _n * _n / 45,
    56 * _n * _n * _n / 15 - 136 * _n * _n * _n * _n / 35,
    4279 * _n * _n * _n * _n / 630,
  ];

  // Returns [latitude, longitude] in degrees.
  static List<double> _utmToLatLng(
      int zone, bool north, double easting, double northing) {
    final double xi = (northing - (north ? 0.0 : 10000000.0)) / (_k0 * _bigA);
    final double eta = (easting - 500000.0) / (_k0 * _bigA);

    double xiP = xi, etaP = eta;
    for (int j = 1; j <= 4; j++) {
      xiP -= _beta[j - 1] * sin(2 * j * xi) * _cosh(2 * j * eta);
      etaP -= _beta[j - 1] * cos(2 * j * xi) * _sinh(2 * j * eta);
    }

    final double chi = asin(sin(xiP) / _cosh(etaP));
    double phi = chi;
    for (int j = 1; j <= 4; j++) {
      phi += _delta[j - 1] * sin(2 * j * chi);
    }

    final double lambda0 = (zone * 6 - 183) * pi / 180;
    final double lambda = lambda0 + atan2(_sinh(etaP), cos(xiP));

    return [phi * 180 / pi, lambda * 180 / pi];
  }
}

// One satellite tile: its UTM zone, hemisphere and code.
class _Tile {
  final int zone;
  final bool north;
  final String id;
  const _Tile(this.zone, this.north, this.id);
}

// All the detections of one debris type inside one map cell.
class _Cluster {
  final String type;
  final String tileId;

  // Positions stored flat as lat, lng, lat, lng ... (no object per point).
  final List<double> _coords = [];

  // The danger level / action are the same for every row of a debris type
  // in this report, but the most common one is used in case they ever differ.
  final Map<String, int> _dangerVotes = {};
  final Map<String, int> _actionVotes = {};

  double _latSum = 0;
  double _lngSum = 0;

  _Cluster(this.type, this.tileId);

  int get count => _coords.length ~/ 2;

  void add(double lat, double lng,
      {required String danger, required String action}) {
    _coords
      ..add(lat)
      ..add(lng);
    _latSum += lat;
    _lngSum += lng;
    final String d = danger.trim();
    final String a = action.trim();
    if (d.isNotEmpty) _dangerVotes[d] = (_dangerVotes[d] ?? 0) + 1;
    if (a.isNotEmpty) _actionVotes[a] = (_actionVotes[a] ?? 0) + 1;
  }

  LatLng get center => LatLng(_latSum / count, _lngSum / count);

  String get danger => _winner(_dangerVotes, 'Low');
  String get action => _winner(_actionVotes, '');

  static String _winner(Map<String, int> votes, String fallback) {
    String best = fallback;
    int bestCount = 0;
    votes.forEach((k, v) {
      if (v > bestCount) {
        best = k;
        bestCount = v;
      }
    });
    return best;
  }

  // The outline of the zone: the convex hull around every detection, each
  // one padded to a small square so the shape never collapses to a line.
  // (Padding the hull's corners gives the same shape as padding every
  // point, with far fewer points to sort.)
  List<LatLng> outline() {
    const double pad = CsvZonesLoader._padDegrees;

    final List<_P> raw = [
      for (int i = 0; i < _coords.length; i += 2) _P(_coords[i + 1], _coords[i]),
    ];
    final List<_P> hull = _convexHull(raw);

    final List<_P> padded = [];
    for (final _P p in hull) {
      padded
        ..add(_P(p.x - pad, p.y - pad))
        ..add(_P(p.x - pad, p.y + pad))
        ..add(_P(p.x + pad, p.y + pad))
        ..add(_P(p.x + pad, p.y - pad));
    }
    return _convexHull(padded).map((p) => LatLng(p.y, p.x)).toList();
  }

  // Andrew's monotone chain.
  static List<_P> _convexHull(List<_P> input) {
    final List<_P> pts = List.of(input)
      ..sort((a, b) => a.x != b.x ? a.x.compareTo(b.x) : a.y.compareTo(b.y));
    if (pts.length < 3) return pts;

    double cross(_P o, _P a, _P b) =>
        (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

    final List<_P> lower = [];
    for (final _P p in pts) {
      while (lower.length >= 2 &&
          cross(lower[lower.length - 2], lower.last, p) <= 0) {
        lower.removeLast();
      }
      lower.add(p);
    }
    final List<_P> upper = [];
    for (final _P p in pts.reversed) {
      while (upper.length >= 2 &&
          cross(upper[upper.length - 2], upper.last, p) <= 0) {
        upper.removeLast();
      }
      upper.add(p);
    }
    lower.removeLast();
    upper.removeLast();
    return [...lower, ...upper];
  }
}

class _P {
  final double x;
  final double y;
  const _P(this.x, this.y);
}
