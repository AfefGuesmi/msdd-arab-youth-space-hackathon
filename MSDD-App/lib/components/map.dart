import 'dart:math';
import 'package:flutter/material.dart';
import 'package:flutter_map/flutter_map.dart';
import 'package:latlong2/latlong.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

import './auth_service.dart';
import './csv_zones_loader.dart';
import './danger_style.dart';
import './popup_widget.dart';
import './reports_sidebar.dart';
import './notifications_sidebar.dart';
import './zones_sidebar.dart';

// Which right-side panel (if any) is currently open. Only one panel can
// be visible at a time — opening one always closes the other. Reports and
// Notifications are fully separate panels.
enum _SidePanel { none, reports, notifications, zones }

// One entry in the circular radial menu that fans out from the main
// bottom-right button.
class _RadialAction {
  final _SidePanel? panel; // null for actions that don't open a panel
  final bool isSignOut; // true for the Sign out action
  final IconData icon;
  final String tooltip;
  final String? label; // small message shown beside the icon (chatbot)

  const _RadialAction({
    required this.panel,
    required this.icon,
    required this.tooltip,
    this.isSignOut = false,
    this.label,
  });
}

class MapPage extends StatefulWidget {
  const MapPage({super.key});

  @override
  State<MapPage> createState() => _MapPageState();
}

class _MapPageState extends State<MapPage> with TickerProviderStateMixin {
  static const String _fontFamily = 'Times New Roman';

  // Same dark navy used across the sidebars, now shared by the app bar
  // and the circular menu buttons so everything reads as one palette.
  static const Color sidebarDarkBlue = Color(0xFF0A0E1A);
  static const Color sidebarDarkBlue2 = Color(0xFF1A1F33);

  static const double _panelWidth = 320;
  static const double _topBarHeight = 56;

  final MapController _mapController = MapController();
  final FlutterLocalNotificationsPlugin _localNotifications =
      FlutterLocalNotificationsPlugin();

  late AnimationController _pulseController;

  // Drives the radial menu's open/close animation (expansion + fade).
  late AnimationController _radialController;

  double _currentZoom = 4;
  // Zone icons only appear once the map is zoomed in this far, and only for
  // the zones inside the visible part of the map (never more than
  // _maxMarkers at once), so the map stays fast even with hundreds of
  // zones. The sidebar always lists every zone.
  final double _markerZoomThreshold = 8;
  static const int _maxMarkers = 60;

  // Zoom used when the map flies to an area (first zone, closing a popup).
  static const double _areaZoom = 9;

  // The part of the world currently visible on screen.
  LatLngBounds? _visibleBounds;

  LatLng _currentCenter = LatLng(36.8065, 10.1815);

  // One notification per detected area: {message, danger, zone}.
  List<Map<String, dynamic>> alerts = [];
  String? activeAlert;
  Map<String, dynamic>? selectedZone;

  // The zone chosen in the Detected Zones sidebar. While it's set, the
  // sidebar shows that zone's info and image instead of the list.
  Map<String, dynamic>? _focusedZone;

  // Only one of these can be open at once, and all three slide in from
  // the right edge of the screen, on top of the app bar.
  _SidePanel _activePanel = _SidePanel.none;

  // Whether the circular menu is expanded.
  bool _radialOpen = false;

  // Extra rotation (in degrees) applied to every icon in the radial menu.
  // Dragging across the main button while it's open adds to this, which
  // sweeps all the sub-icons around the main button in a circular motion.
  double _radialRotation = 0;

  // Position of the main circular button, expressed as distance from the
  // right/bottom edges. Starts at its original bottom-right spot; a long
  // press lets the user drag it anywhere on screen.
  double _fabRight = 20;
  double _fabBottom = 24;
  Offset? _dragStartGlobal;
  double _dragStartRight = 20;
  double _dragStartBottom = 24;

  // The five actions that fan out of the main button. The chatbot is only
  // an icon with its "Ask MSDD Assistant" message for now (no page yet). The last one
  // signs the person out and returns to the home page.
  final List<_RadialAction> _radialActions = const [
    _RadialAction(
      panel: _SidePanel.reports,
      icon: Icons.description_outlined,
      tooltip: "Reports",
    ),
    _RadialAction(
      panel: _SidePanel.notifications,
      icon: Icons.notifications_outlined,
      tooltip: "Notifications",
    ),
    _RadialAction(
      panel: _SidePanel.zones,
      icon: Icons.location_on_outlined,
      tooltip: "Detected zones",
    ),
    _RadialAction(
      panel: null,
      icon: Icons.smart_toy_outlined,
      tooltip: "Assistant",
      label: "Ask MSDD Assistant",
    ),
    _RadialAction(
      panel: null,
      isSignOut: true,
      icon: Icons.logout,
      tooltip: "Sign out",
    ),
  ];

  // The detected zones, built from assets/files/mssd_debris_report.csv when
  // the page opens (see csv_zones_loader.dart).
  List<Map<String, dynamic>> debrisZones = [];
  bool _zonesLoading = true;
  String? _zonesError;

  @override
  void initState() {
    super.initState();

    _initNotifications();

    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(seconds: 2),
    )..repeat();

    _radialController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 260),
    );

    _loadZones();
  }

  // Reads the CSV, then builds the first notifications from what it found.
  Future<void> _loadZones() async {
    setState(() {
      _zonesLoading = true;
      _zonesError = null;
    });

    try {
      final zones = await CsvZonesLoader.load();
      if (!mounted) return;
      setState(() {
        debrisZones = zones;
        alerts = zones.map(_alertFor).toList();
        _zonesLoading = false;
        if (zones.isEmpty) {
          _zonesError =
              "The zones file was found but no valid rows could be read from it.";
        }
      });

      // The report covers several seas far apart, so start the map on the
      // first (most severe, largest) zone instead of an empty ocean.
      if (zones.isNotEmpty) {
        final LatLng first = zones.first["center"] as LatLng;
        try {
          _mapController.move(first, _areaZoom);
        } catch (_) {
          // The map isn't ready yet — it simply stays where it is.
        }
      }
    } catch (e, st) {
      debugPrint("Zones file error: $e\n$st");
      if (!mounted) return;
      setState(() {
        _zonesLoading = false;
        // The loader's own "no CSV found" message already says what to fix;
        // anything else gets the generic text plus the real cause.
        final String detail = e.toString();
        _zonesError = detail.startsWith('No .csv file found')
            ? detail
            : "Couldn't read the detected zones file. ($detail)";
      });
    }
  }

  // High = red, Medium = orange, normal (Low) = yellow.
  Color _dangerColor(String level) => DangerStyle.color(level);

  // The notification for one detected area, worded and coloured by its
  // danger level ("Alert", "Warning" or "Notice").
  Map<String, dynamic> _alertFor(Map<String, dynamic> zone) {
    final LatLng c = zone["center"] as LatLng;
    final String danger = zone["danger"] as String;
    return {
      "message":
          "${DangerStyle.label(danger)}: ${zone["type"]} detected in ${zone["ocean"]} (${c.latitude.toStringAsFixed(3)}, ${c.longitude.toStringAsFixed(3)})",
      "danger": danger,
      "zone": zone,
    };
  }

  double _markerSize() {
    return (_currentZoom * 3).clamp(6, 26);
  }

  Widget _buildMarker(Map<String, dynamic> zone) {
    final color = _dangerColor(zone["danger"]);
    final size = _markerSize();

    return AnimatedBuilder(
      animation: _pulseController,
      builder: (context, child) {
        double pulse = 1 + (_pulseController.value * 0.8);

        return Stack(
          alignment: Alignment.center,
          children: [
            Container(
              width: size * pulse,
              height: size * pulse,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: color.withOpacity(0.15),
              ),
            ),
            Container(
              width: size,
              height: size,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: color,
                border: Border.all(color: Colors.white, width: 1.5),
              ),
            ),
          ],
        );
      },
    );
  }

  Future<void> _initNotifications() async {
    const AndroidInitializationSettings androidInit =
        AndroidInitializationSettings('@mipmap/ic_launcher');

    const InitializationSettings settings = InitializationSettings(
      android: androidInit,
    );

    await _localNotifications.initialize(settings);
  }

  void _sendNotification(String message) async {
    const AndroidNotificationDetails androidDetails =
        AndroidNotificationDetails(
      'alerts_channel',
      'Alerts',
      importance: Importance.max,
      priority: Priority.high,
    );

    const NotificationDetails details = NotificationDetails(
      android: androidDetails,
    );

    await _localNotifications.show(0, 'New Alert', message, details);
  }

  void _addAlert(Map<String, dynamic> zone) {
    final Map<String, dynamic> alert = _alertFor(zone);
    final String message = alert["message"] as String;

    setState(() {
      alerts.insert(0, alert);
      activeAlert = message;
    });

    _sendNotification(message);

    Future.delayed(const Duration(seconds: 5), () {
      if (mounted) {
        setState(() {
          activeAlert = null;
        });
      }
    });
  }

  // Zooms in on a zone and keeps the sidebar open so it can show the zone's
  // information and image. The sidebar covers the right part of the screen,
  // so the map is centred a little to the east of the zone — that lands the
  // zone in the middle of the part of the map that is still visible,
  // instead of hiding it behind the sidebar.
  void _focusZone(Map<String, dynamic> zone) {
    const double targetZoom = 10;
    final LatLng center = zone["center"] as LatLng;
    final double screenWidth = MediaQuery.of(context).size.width;
    final double coveredPx = min(_panelWidth, screenWidth);

    // At web-mercator zoom z, one pixel is 360 / (256 * 2^z) degrees of
    // longitude. Half the sidebar width in pixels, converted to degrees:
    final double shiftDegrees = (coveredPx / 2) * 360 / (256 * pow(2, targetZoom));
    final LatLng target = LatLng(center.latitude, center.longitude + shiftDegrees);

    setState(() {
      _focusedZone = zone;
      _currentCenter = target;
      _currentZoom = targetZoom;
    });
    _mapController.move(target, targetZoom);
  }

  // How close (in screen pixels) a tap must be to a zone's marker to count
  // as a tap on that zone — small zones are only a few pixels wide.
  static const double _tapRadiusPx = 28;

  bool _tapHitsZone(LatLng point, Map<String, dynamic> zone) {
    final polygon = (zone["polygon"] as List).cast<LatLng>();
    if (_pointInPolygon(point, polygon)) return true;

    final LatLng c = zone["center"] as LatLng;
    final double cosLat = cos(c.latitude * pi / 180);
    final double degPerPx = 360 / (256 * pow(2, _currentZoom));
    final double dLat = point.latitude - c.latitude;
    final double dLng = (point.longitude - c.longitude) * cosLat;
    return sqrt(dLat * dLat + dLng * dLng) <= _tapRadiusPx * degPerPx * cosLat;
  }

  // The zones that get an icon on the map right now:
  //  - a zone picked in the sidebar: just that one;
  //  - zoomed out: none;
  //  - zoomed in: only the zones inside the visible map area, and if there
  //    are still too many, the ones closest to the middle of the screen.
  List<Map<String, dynamic>> _zonesInView() {
    final Map<String, dynamic>? focused = _focusedZone;
    if (focused != null) return [focused];
    if (_currentZoom < _markerZoomThreshold) return <Map<String, dynamic>>[];

    // The visible area plus a margin of 20% on each side, so icons are
    // already there when they slide into view while the map is moved.
    final LatLngBounds? bounds = _visibleBounds;
    final List<Map<String, dynamic>> inView;
    if (bounds == null) {
      inView = List.of(debrisZones);
    } else {
      final double padLat = (bounds.north - bounds.south) * 0.2;
      final double padLng = (bounds.east - bounds.west) * 0.2;
      inView = debrisZones.where((z) {
        final LatLng c = z["center"] as LatLng;
        return c.latitude >= bounds.south - padLat &&
            c.latitude <= bounds.north + padLat &&
            c.longitude >= bounds.west - padLng &&
            c.longitude <= bounds.east + padLng;
      }).toList();
    }

    if (inView.length > _maxMarkers) {
      double dist(Map<String, dynamic> z) {
        final LatLng c = z["center"] as LatLng;
        final double dLat = c.latitude - _currentCenter.latitude;
        final double dLng = c.longitude - _currentCenter.longitude;
        return dLat * dLat + dLng * dLng;
      }

      inView.sort((a, b) => dist(a).compareTo(dist(b)));
      return inView.sublist(0, _maxMarkers);
    }
    return inView;
  }

  // Closing the popup flies back out to the zone's area (so its neighbours
  // are visible too) and shows the icons of the zones around it.
  void _closePopup() {
    final Map<String, dynamic>? zone = selectedZone;
    setState(() {
      selectedZone = null;
      _focusedZone = null;
    });
    if (zone != null) {
      final LatLng center = zone["center"] as LatLng;
      _mapController.move(center, _areaZoom);
    }
  }

  void _handleMapTap(LatLng point) {
    final Map<String, dynamic>? isolated = _focusedZone;

    // A zone is picked in the sidebar and is the only one on the map.
    if (isolated != null) {
      if (_tapHitsZone(point, isolated)) {
        // Tapping it opens its popup and closes the sidebar (the zone
        // stays alone on the map).
        setState(() {
          selectedZone = isolated;
          _activePanel = _SidePanel.none;
        });
      } else if (_activePanel == _SidePanel.none) {
        // Sidebar already closed: tapping elsewhere brings all zones back.
        setState(() => _focusedZone = null);
      }
      return;
    }

    for (final zone in _zonesInView()) {
      if (_tapHitsZone(point, zone)) {
        setState(() {
          selectedZone = zone;
        });
        break;
      }
    }
  }

  // Used by the notifications: opens the Detected Zones sidebar on that
  // zone's information and zooms the map to the zone (the same thing that
  // happens when the zone is picked from the zones list).
  void _showZoneFromNotification(Map<String, dynamic> zone) {
    setState(() => _activePanel = _SidePanel.zones);
    _focusZone(zone);
  }

  // Opening a panel always closes the other ones, since all three now
  // live on the same right edge.
  void _handleRadialAction(_RadialAction action) {
    if (action.isSignOut) {
      setState(() {
        _radialOpen = false;
        _radialController.reverse();
      });
      AuthService.instance.signOut();
      // Back to the home page, and the map can't be reached with "back".
      Navigator.pushNamedAndRemoveUntil(context, '/', (route) => false);
      return;
    }
    // Actions without a panel (the chatbot, for now) just fold the menu.
    if (action.panel == null) {
      setState(() {
        _radialOpen = false;
        _radialController.reverse();
      });
      return;
    }
    _openPanel(action.panel);
  }

  void _openPanel(_SidePanel? panel) {
    setState(() {
      _radialOpen = false;
      _radialController.reverse();
      // Reopening (or switching) a panel always starts from the zones list.
      _focusedZone = null;
      if (panel == null) {
        // Action with no panel — just close the menu.
        return;
      }
      _activePanel = _activePanel == panel ? _SidePanel.none : panel;
    });
  }

  void _closePanels() {
    setState(() {
      _activePanel = _SidePanel.none;
      _focusedZone = null;
    });
  }

  void _toggleRadialMenu() {
    setState(() {
      _radialOpen = !_radialOpen;
      if (_radialOpen) {
        _radialController.forward();
      } else {
        _radialController.reverse();
      }
    });
  }

  // --- Dragging the main button to a new spot (long-press to grab) ---

  void _handleLongPressStart(LongPressStartDetails details) {
    setState(() {
      _dragStartGlobal = details.globalPosition;
      _dragStartRight = _fabRight;
      _dragStartBottom = _fabBottom;
    });
  }

  void _handleLongPressMoveUpdate(
      LongPressMoveUpdateDetails details, Size screenSize, double mainSize) {
    if (_dragStartGlobal == null) return;
    final delta = details.globalPosition - _dragStartGlobal!;

    double newRight = _dragStartRight - delta.dx;
    double newBottom = _dragStartBottom - delta.dy;

    // Keep the button fully on screen.
    final double maxRight = screenSize.width - mainSize - 4;
    final double maxBottom = screenSize.height - mainSize - 4;
    newRight = newRight.clamp(4, maxRight < 4 ? 4 : maxRight);
    newBottom = newBottom.clamp(4, maxBottom < 4 ? 4 : maxBottom);

    setState(() {
      _fabRight = newRight;
      _fabBottom = newBottom;
    });
  }

  void _handleLongPressEnd(LongPressEndDetails details) {
    setState(() {
      _dragStartGlobal = null;
    });
  }

  @override
  void dispose() {
    _pulseController.dispose();
    _radialController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final bool panelOpen = _activePanel != _SidePanel.none;
    // While a zone is shown in the sidebar, the dimming scrim is removed so
    // the zoomed-in area stays clearly visible (and the map stays usable).
    final bool scrimVisible = panelOpen && _focusedZone == null;
    final Size screenSize = MediaQuery.of(context).size;

    // The zones drawn on the map (see _zonesInView).
    final List<Map<String, dynamic>> visibleZones = _zonesInView();

    return Scaffold(
      // No built-in AppBar anymore — the top bar is drawn as a Stack
      // layer instead, so the sliding panels (which come later in the
      // Stack) can render above it and start right from the top edge
      // of the screen instead of stopping underneath it.
      body: Stack(
        children: [
          FlutterMap(
            mapController: _mapController,
            options: MapOptions(
              initialCenter: _currentCenter,
              initialZoom: _currentZoom,
              onTap: (tapPosition, point) => _handleMapTap(point),
              onPositionChanged: (position, hasGesture) {
                setState(() {
                  _currentCenter = position.center;
                  _currentZoom = position.zoom;
                  _visibleBounds = position.visibleBounds;

                  // Dragging or zooming the map by hand while a single zone
                  // is isolated (and the sidebar is closed) releases it, so
                  // the icons of whatever area you move into load in.
                  if (hasGesture &&
                      _focusedZone != null &&
                      _activePanel == _SidePanel.none) {
                    _focusedZone = null;
                  }
                });
              },
            ),
            children: [
              // CartoDB's free anonymous basemap tiles now require an
              // account/API key for most usage, which is why the map
              // stopped loading. Esri's dark-gray canvas basemap is free
              // and needs no key — two layers stacked: a plain dark base,
              // then a semi-transparent "reference" layer with place
              // labels/borders on top of it.
              TileLayer(
                urlTemplate:
                    'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}',
                userAgentPackageName: 'com.msdd.app',
                maxNativeZoom: 16,
              ),
              TileLayer(
                urlTemplate:
                    'https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}',
                userAgentPackageName: 'com.msdd.app',
                maxNativeZoom: 16,
              ),
              // While a zone is picked in the sidebar, only that zone is
              // kept on the map: its outline is drawn and every other
              // zone is hidden.
              if (_focusedZone != null)
                PolygonLayer(
                  polygons: [
                    Polygon(
                      points: (_focusedZone!["polygon"] as List).cast<LatLng>(),
                      color: _dangerColor(_focusedZone!["danger"] as String)
                          .withOpacity(0.28),
                      borderColor:
                          _dangerColor(_focusedZone!["danger"] as String),
                      borderStrokeWidth: 2,
                    ),
                  ],
                ),
              MarkerLayer(
                markers: visibleZones.map((zone) {
                  return Marker(
                    point: zone["center"] as LatLng,
                    width: 60,
                    height: 60,
                    child: _buildMarker(zone),
                  );
                }).toList(),
              ),
              // Esri requires attribution for this basemap — shown as a
              // small expandable "i" button in the map's corner.
              RichAttributionWidget(
                alignment: AttributionAlignment.bottomLeft,
                attributions: [
                  TextSourceAttribution(
                    'Esri, HERE, Garmin, © OpenStreetMap contributors',
                  ),
                ],
              ),
            ],
          ),
          // Custom top bar. "Map" is explicitly white, and this whole bar
          // sits low enough in the Stack that the panels can slide in
          // above it.
          Positioned(
            top: 0,
            left: 0,
            right: 0,
            child: Container(
              color: sidebarDarkBlue,
              child: SafeArea(
                bottom: false,
                child: SizedBox(
                  height: _topBarHeight,
                  child: const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 16),
                    child: Align(
                      alignment: Alignment.centerLeft,
                      child: Text(
                        "Map",
                        style: TextStyle(
                          color: Colors.white,
                          fontFamily: _fontFamily,
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 0.5,
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
          if (_zonesLoading)
            Positioned(
              top: _topBarHeight + 12,
              left: 0,
              right: 0,
              child: SafeArea(
                child: Center(
                  child: Container(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 14, vertical: 10),
                    decoration: BoxDecoration(
                      color: sidebarDarkBlue.withOpacity(0.92),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: const Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        SizedBox(
                          width: 14,
                          height: 14,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            valueColor: AlwaysStoppedAnimation(Colors.white),
                          ),
                        ),
                        SizedBox(width: 10),
                        Text(
                          "Loading detected zones...",
                          style: TextStyle(
                            color: Colors.white,
                            fontFamily: _fontFamily,
                            fontSize: 12,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          if (_zonesError != null)
            Positioned(
              top: _topBarHeight + 12,
              left: 20,
              right: 20,
              child: SafeArea(
                child: Container(
                  padding: const EdgeInsets.symmetric(
                      horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: Colors.red.shade900.withOpacity(0.95),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: Row(
                    children: [
                      const Icon(Icons.error_outline,
                          color: Colors.white, size: 18),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Text(
                          _zonesError!,
                          style: const TextStyle(
                            color: Colors.white,
                            fontFamily: _fontFamily,
                            fontSize: 12,
                          ),
                        ),
                      ),
                      TextButton(
                        onPressed: _loadZones,
                        child: const Text(
                          "Retry",
                          style: TextStyle(
                            color: Colors.white,
                            fontFamily: _fontFamily,
                            fontWeight: FontWeight.bold,
                            fontSize: 12,
                          ),
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ),
          if (selectedZone != null)
            Center(
              child: PopupWidget(
                debrisType: selectedZone!["type"],
                dangerLevel: selectedZone!["danger"],
                solution: selectedZone!["solution"],
                coords: selectedZone!["center"],
                imagePath: selectedZone!["image"] as String?,
                onClose: _closePopup,
              ),
            ),
          if (activeAlert != null)
            Positioned(
              top: _topBarHeight + 12,
              left: 20,
              right: 20,
              child: Container(
                padding: const EdgeInsets.symmetric(
                  vertical: 14,
                  horizontal: 12,
                ),
                decoration: BoxDecoration(
                  color: Colors.red.withOpacity(0.5),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Row(
                  children: [
                    const Icon(Icons.warning, color: Colors.white),
                    const SizedBox(width: 10),
                    Expanded(
                      child: Text(
                        activeAlert!,
                        style: const TextStyle(
                          color: Colors.white,
                          fontFamily: _fontFamily,
                          fontSize: 13,
                        ),
                      ),
                    ),
                    GestureDetector(
                      onTap: () => setState(() => activeAlert = null),
                      child: const Icon(Icons.close, color: Colors.white),
                    ),
                  ],
                ),
              ),
            ),
          // Dark scrim behind whichever panel is open, tap to dismiss.
          IgnorePointer(
            ignoring: !scrimVisible,
            child: AnimatedOpacity(
              duration: const Duration(milliseconds: 250),
              opacity: scrimVisible ? 1 : 0,
              child: GestureDetector(
                onTap: _closePanels,
                child: Container(color: Colors.black.withOpacity(0.45)),
              ),
            ),
          ),
          // The three panels start from the very top of the screen (top:
          // 0) and are stacked after the top bar above, so they render on
          // top of it whenever they're open — not tucked underneath.
          _buildSlidingPanel(
            visible: _activePanel == _SidePanel.reports,
            title: "Reports",
            child: ReportsSidebar(zones: debrisZones),
          ),
          _buildSlidingPanel(
            visible: _activePanel == _SidePanel.notifications,
            title: "Notifications",
            child: NotificationsSidebar(
              notifications: alerts,
              onDismissNotification: (index) {
                setState(() {
                  alerts.removeAt(index);
                });
              },
              // Tapping a notification zooms to that area and shows its
              // information in the zones sidebar.
              onNotificationTap: _showZoneFromNotification,
            ),
          ),
          _buildSlidingPanel(
            visible: _activePanel == _SidePanel.zones,
            title: "Detected Zones",
            child: ZonesSidebar(
              zones: debrisZones,
              selectedZone: _focusedZone,
              // Zoom the map to the zone and keep the sidebar open: it
              // switches to that zone's info + image.
              onZoneSelected: _focusZone,
              onBack: () => setState(() => _focusedZone = null),
            ),
          ),
          // Scrim behind the radial menu itself, so tapping anywhere
          // outside the fanned-out icons collapses it back down.
          IgnorePointer(
            ignoring: !_radialOpen,
            child: GestureDetector(
              onTap: _toggleRadialMenu,
              behavior: HitTestBehavior.translucent,
              child: Container(color: Colors.transparent),
            ),
          ),
          _buildRadialMenu(screenSize),
        ],
      ),
    );
  }

  // Shared "chrome" for the three right-side panels: a title bar with a
  // close button, then the panel's own content underneath. No rounded
  // corners anymore — straight edges, flush with the screen edge — and
  // it spans the full screen height starting at the very top.
  Widget _buildSlidingPanel({
    required bool visible,
    required String title,
    required Widget child,
  }) {
    return AnimatedPositioned(
      duration: const Duration(milliseconds: 300),
      curve: Curves.easeInOutCubic,
      top: 0,
      bottom: 0,
      right: visible ? 0 : -_panelWidth,
      width: _panelWidth,
      child: Material(
        color: Colors.transparent,
        child: Container(
          decoration: const BoxDecoration(
            gradient: LinearGradient(
              colors: [sidebarDarkBlue, sidebarDarkBlue2],
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
            ),
            boxShadow: [
              BoxShadow(
                color: Colors.black54,
                blurRadius: 24,
                offset: Offset(-6, 0),
              ),
            ],
          ),
          child: SafeArea(
            child: Column(
              children: [
                Padding(
                  padding: const EdgeInsets.fromLTRB(20, 14, 8, 10),
                  child: Row(
                    children: [
                      Expanded(
                        child: Text(
                          title,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 15,
                            fontWeight: FontWeight.bold,
                            fontFamily: _fontFamily,
                          ),
                        ),
                      ),
                      IconButton(
                        icon: const Icon(Icons.close,
                            color: Colors.white70, size: 20),
                        onPressed: _closePanels,
                      ),
                    ],
                  ),
                ),
                const Divider(color: Colors.white12, height: 1),
                Expanded(child: child),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // The circular menu: a main button that fans its four actions out
  // along a quarter-circle arc when tapped. Dragging across the main
  // button while it's open sweeps the whole fan around in a circular
  // motion, like turning a small dial. Long-pressing the main button
  // while it's closed lets the user drag it anywhere on screen; it
  // starts out in its original bottom-right spot.
  Widget _buildRadialMenu(Size screenSize) {
    const double mainSize = 58;
    const double subSize = 44;
    // Five icons share the quarter circle, so the arc is a bit wider to
    // keep them from touching.
    const double radius = 124;

    return AnimatedBuilder(
      animation: _radialController,
      builder: (context, _) {
        final double t = Curves.easeOutBack.transform(_radialController.value);
        final double fadeT = _radialController.value;
        final int count = _radialActions.length;

        return Stack(
          clipBehavior: Clip.none,
          children: [
            // Sub-actions, fanned from straight up (0°) to straight left
            // (90°) around the main button, plus whatever extra rotation
            // has been dragged in.
            for (int i = 0; i < count; i++)
              Builder(builder: (context) {
                final double spread = count > 1 ? 90 / (count - 1) : 0;
                final double baseAngleDeg = i * spread;
                final double angleRad =
                    (baseAngleDeg + _radialRotation) * pi / 180;
                final double dist = radius * t;
                final double dx = dist * sin(angleRad);
                final double dy = dist * cos(angleRad);
                final action = _radialActions[i];
                final bool isActive = action.panel != null &&
                    _activePanel == action.panel;

                return Positioned(
                  right: _fabRight + dx - (subSize - mainSize) / 2,
                  bottom: _fabBottom + dy - (subSize - mainSize) / 2,
                  child: Opacity(
                    opacity: fadeT.clamp(0, 1),
                    child: Transform.scale(
                      scale: 0.6 + (0.4 * t),
                      // The button stays pinned to the right edge of this row,
                      // so a message (like the assistant's label) sits to
                      // its left without moving the button itself.
                      child: Row(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          if (action.label != null) ...[
                            Container(
                              padding: const EdgeInsets.symmetric(
                                  horizontal: 10, vertical: 5),
                              decoration: BoxDecoration(
                                color: Colors.white,
                                borderRadius: BorderRadius.circular(14),
                                boxShadow: [
                                  BoxShadow(
                                    color: Colors.black.withOpacity(0.3),
                                    blurRadius: 6,
                                    offset: const Offset(0, 2),
                                  ),
                                ],
                              ),
                              child: Text(
                                action.label!,
                                style: const TextStyle(
                                  color: sidebarDarkBlue,
                                  fontFamily: _fontFamily,
                                  fontSize: 11,
                                  fontWeight: FontWeight.bold,
                                ),
                              ),
                            ),
                            const SizedBox(width: 8),
                          ],
                          _RadialButton(
                            size: subSize,
                            iconSize: 18,
                            icon: action.icon,
                            tooltip: action.tooltip,
                            highlighted: isActive,
                            onTap: () => _handleRadialAction(action),
                          ),
                        ],
                      ),
                    ),
                  ),
                );
              }),
            // Main circular button, always on top. Tap toggles the fan.
            // Drag (while open) rotates the fan. Long-press (while
            // closed) grabs the button so it can be dragged anywhere.
            Positioned(
              right: _fabRight,
              bottom: _fabBottom,
              child: GestureDetector(
                onTap: _toggleRadialMenu,
                onPanUpdate: _radialOpen
                    ? (details) {
                        setState(() {
                          _radialRotation += details.delta.dx * 0.8;
                        });
                      }
                    : null,
                onLongPressStart:
                    _radialOpen ? null : _handleLongPressStart,
                onLongPressMoveUpdate: _radialOpen
                    ? null
                    : (details) => _handleLongPressMoveUpdate(
                        details, screenSize, mainSize),
                onLongPressEnd: _radialOpen ? null : _handleLongPressEnd,
                child: _RadialButton(
                  size: mainSize,
                  iconSize: 24,
                  icon: _radialOpen ? Icons.close : Icons.apps_rounded,
                  tooltip: _radialOpen ? "Close menu" : "Menu",
                  highlighted: false,
                  onTap: _toggleRadialMenu,
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  bool _pointInPolygon(LatLng point, List<LatLng> polygon) {
    int intersectCount = 0;
    // Every edge, including the last one back to the first point.
    for (int j = 0; j < polygon.length; j++) {
      if (rayCastIntersect(point, polygon[j], polygon[(j + 1) % polygon.length])) {
        intersectCount++;
      }
    }
    return (intersectCount % 2) == 1;
  }

  bool rayCastIntersect(LatLng point, LatLng vertA, LatLng vertB) {
    final px = point.longitude;
    final py = point.latitude;
    final ax = vertA.longitude;
    final ay = vertA.latitude;
    final bx = vertB.longitude;
    final by = vertB.latitude;

    if ((ay > py && by > py) || (ay < py && by < py) || (ax < px && bx < px)) {
      return false;
    }

    final m = (ay != by) ? (bx - ax) / (by - ay) : double.infinity;
    final x = m * (py - ay) + ax;
    return x > px;
  }
}

// A small circular icon button shared by the main radial button and all
// of its fanned-out sub-actions, so their size/shape stay consistent.
// Filled with the same dark navy blue used across the sidebars.
class _RadialButton extends StatelessWidget {
  final double size;
  final double iconSize;
  final IconData icon;
  final String tooltip;
  final bool highlighted;
  final VoidCallback onTap;

  const _RadialButton({
    required this.size,
    required this.iconSize,
    required this.icon,
    required this.tooltip,
    required this.onTap,
    this.highlighted = false,
  });

  static const Color darkBlue = _MapPageState.sidebarDarkBlue;

  @override
  Widget build(BuildContext context) {
    final Color bg = highlighted ? Colors.white : darkBlue;
    final Color fg = highlighted ? darkBlue : Colors.white;

    return Tooltip(
      message: tooltip,
      child: Material(
        color: Colors.transparent,
        shape: const CircleBorder(),
        child: InkWell(
          customBorder: const CircleBorder(),
          onTap: onTap,
          child: Container(
            width: size,
            height: size,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: bg,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withOpacity(0.35),
                  blurRadius: 8,
                  offset: const Offset(0, 3),
                ),
              ],
            ),
            child: Icon(icon, color: fg, size: iconSize),
          ),
        ),
      ),
    );
  }
}