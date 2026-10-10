import 'package:flutter/material.dart';
import 'package:latlong2/latlong.dart';

import './danger_style.dart';
import './zone_image.dart';

/// Two views in one sidebar:
///  • the list of detected zones (nothing selected), and
///  • the detail view of one zone — its image and information — which
///    appears after the map has zoomed to that zone.
/// The selected zone is owned by map.dart, so the sidebar stays open
/// while the map zooms and the map can react to it.
class ZonesSidebar extends StatelessWidget {
  final List<Map<String, dynamic>> zones;
  final Map<String, dynamic>? selectedZone;
  final ValueChanged<Map<String, dynamic>> onZoneSelected;
  final VoidCallback onBack;

  const ZonesSidebar({
    super.key,
    required this.zones,
    required this.selectedZone,
    required this.onZoneSelected,
    required this.onBack,
  });

  static const String _fontFamily = 'Times New Roman';

  static Color _dangerColor(String level) => DangerStyle.color(level);

  @override
  Widget build(BuildContext context) {
    // No Drawer wrapper here — this widget is embedded directly inside
    // the sliding right-side panel built in map.dart, which already
    // provides the background and title bar.
    final zone = selectedZone;
    return AnimatedSwitcher(
      duration: const Duration(milliseconds: 220),
      child: zone == null
          ? KeyedSubtree(
              key: const ValueKey('zones-list'),
              child: _buildList(),
            )
          : KeyedSubtree(
              key: ValueKey('zone-detail-${identityHashCode(zone)}'),
              child: _buildDetail(context, zone),
            ),
    );
  }

  // ---------------------------------------------------------------
  // List view
  // ---------------------------------------------------------------
  Widget _buildList() {
    return ListView.builder(
      padding: const EdgeInsets.symmetric(vertical: 4),
      itemCount: zones.length,
      itemBuilder: (context, index) {
        final zone = zones[index];
        final center = zone["center"] as LatLng;
        final dangerColor = _dangerColor(zone["danger"] as String);

        return ListTile(
          dense: true,
          leading: CircleAvatar(
            backgroundColor: dangerColor,
            radius: 6,
          ),
          title: Text(
            zone["type"],
            style: const TextStyle(
              color: Colors.white,
              fontFamily: _fontFamily,
              fontSize: 13,
            ),
          ),
          subtitle: Text(
            "${center.latitude.toStringAsFixed(2)}, ${center.longitude.toStringAsFixed(2)}",
            style: const TextStyle(
              color: Colors.white70,
              fontFamily: _fontFamily,
              fontSize: 11,
            ),
          ),
          trailing: const Icon(Icons.zoom_in, color: Colors.white, size: 20),
          // The parent zooms the map to this zone and keeps the sidebar
          // open, which then switches to this zone's detail view.
          onTap: () => onZoneSelected(zone),
        );
      },
    );
  }

  // ---------------------------------------------------------------
  // Detail view
  // ---------------------------------------------------------------
  Widget _buildDetail(BuildContext context, Map<String, dynamic> zone) {
    final center = zone["center"] as LatLng;
    final danger = zone["danger"] as String;
    final dangerColor = _dangerColor(danger);
    final String? imagePath = zone["image"] as String?;

    return ListView(
      padding: const EdgeInsets.fromLTRB(16, 4, 16, 28),
      children: [
        Align(
          alignment: Alignment.centerLeft,
          child: TextButton.icon(
            onPressed: onBack,
            style: TextButton.styleFrom(
              foregroundColor: Colors.white70,
              padding: const EdgeInsets.symmetric(horizontal: 4),
            ),
            icon: const Icon(Icons.arrow_back_ios_new, size: 13),
            label: const Text(
              "All zones",
              style: TextStyle(fontFamily: _fontFamily, fontSize: 12),
            ),
          ),
        ),
        const SizedBox(height: 4),
        _sectionLabel("Qualitative result"),
        const SizedBox(height: 8),
        // Tap the image to open it full size and pinch to zoom.
        GestureDetector(
          onTap: imagePath == null
              ? null
              : () => _openViewer(context, imagePath),
          child: Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.white.withOpacity(0.1)),
            ),
            child: Stack(
              children: [
                ZoneImage(
                  path: imagePath,
                  width: double.infinity,
                  fit: BoxFit.fitWidth,
                  radius: 12,
                ),
                if (imagePath != null)
                  Positioned(
                    right: 8,
                    bottom: 8,
                    child: Container(
                      padding: const EdgeInsets.all(5),
                      decoration: BoxDecoration(
                        shape: BoxShape.circle,
                        color: Colors.black.withOpacity(0.55),
                      ),
                      child: const Icon(Icons.open_in_full,
                          color: Colors.white, size: 13),
                    ),
                  ),
              ],
            ),
          ),
        ),
        const SizedBox(height: 18),
        Row(
          children: [
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                color: dangerColor.withOpacity(0.16),
                border: Border.all(color: dangerColor.withOpacity(0.6)),
              ),
              child: Icon(Icons.warning_rounded, color: dangerColor, size: 16),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                zone["type"],
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.bold,
                  fontFamily: _fontFamily,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 14),
        Divider(color: Colors.white.withOpacity(0.08), height: 1),
        const SizedBox(height: 16),
        _infoRow("Ocean", zone["ocean"] ?? "-"),
        _infoRow("Danger level", danger, valueColor: dangerColor),
        _infoRow(
          "Coordinates",
          "${center.latitude.toStringAsFixed(4)}, ${center.longitude.toStringAsFixed(4)}",
        ),
        _infoRow("Recommended action", zone["solution"] ?? "-"),
      ],
    );
  }

  Widget _sectionLabel(String text) {
    return Text(
      text.toUpperCase(),
      style: const TextStyle(
        color: Colors.white38,
        fontSize: 9.5,
        letterSpacing: 1.1,
        fontFamily: _fontFamily,
        fontWeight: FontWeight.w600,
      ),
    );
  }

  // Same "label above value" style as the map popup.
  Widget _infoRow(String label, String value, {Color? valueColor}) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 14),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _sectionLabel(label),
          const SizedBox(height: 3),
          Text(
            value,
            style: TextStyle(
              color: valueColor ?? Colors.white,
              fontSize: 13,
              fontFamily: _fontFamily,
              fontWeight: FontWeight.w500,
            ),
          ),
        ],
      ),
    );
  }

  // Full-screen, pinch-to-zoom viewer for the zone's image.
  void _openViewer(BuildContext context, String imagePath) {
    showDialog(
      context: context,
      builder: (dialogContext) => Dialog(
        backgroundColor: Colors.black,
        insetPadding: const EdgeInsets.all(12),
        child: Stack(
          children: [
            InteractiveViewer(
              minScale: 1,
              maxScale: 5,
              child: Center(
                child: ZoneImage(
                  path: imagePath,
                  fit: BoxFit.contain,
                  radius: 0,
                ),
              ),
            ),
            Positioned(
              top: 4,
              right: 4,
              child: IconButton(
                icon: const Icon(Icons.close, color: Colors.white),
                onPressed: () => Navigator.of(dialogContext).pop(),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
