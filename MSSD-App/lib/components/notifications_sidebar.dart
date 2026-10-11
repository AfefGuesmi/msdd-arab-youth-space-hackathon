import 'package:flutter/material.dart';

import './danger_style.dart';

// Notifications live in their own sliding panel. There is one notification
// for every detected area, coloured by its danger level:
//   High    -> red alert
//   Medium  -> orange warning
//   Low     -> yellow notice
// Each notification is a map: {message, danger, zone}. Tapping one hands
// its zone back to map.dart, which zooms the map to that area.
class NotificationsSidebar extends StatelessWidget {
  final List<Map<String, dynamic>> notifications;
  final Function(int) onDismissNotification;
  final void Function(Map<String, dynamic> zone) onNotificationTap;

  const NotificationsSidebar({
    Key? key,
    required this.notifications,
    required this.onDismissNotification,
    required this.onNotificationTap,
  }) : super(key: key);

  static const String _fontFamily = 'Times New Roman';

  @override
  Widget build(BuildContext context) {
    if (notifications.isEmpty) {
      return const Center(
        child: Text(
          "No notifications",
          style: TextStyle(
            color: Colors.white70,
            fontFamily: _fontFamily,
            fontSize: 13,
          ),
        ),
      );
    }
    return ListView.builder(
      padding: const EdgeInsets.symmetric(vertical: 4),
      itemCount: notifications.length,
      itemBuilder: (context, index) {
        final Map<String, dynamic> notif = notifications[index];
        final String message = notif["message"] as String;
        final String danger = notif["danger"] as String;
        final Color color = DangerStyle.color(danger);

        return Dismissible(
          key: ValueKey('$message#$index'),
          direction: DismissDirection.endToStart,
          onDismissed: (_) => onDismissNotification(index),
          background: Container(
            alignment: Alignment.centerRight,
            padding: const EdgeInsets.symmetric(horizontal: 20),
            margin: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
            decoration: BoxDecoration(
              color: Colors.red.shade900,
              borderRadius: BorderRadius.circular(8),
            ),
            child: const Icon(Icons.delete, color: Colors.white),
          ),
          child: Card(
            color: color.withOpacity(0.14),
            margin: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
              side: BorderSide(color: color.withOpacity(0.55)),
            ),
            child: ListTile(
              dense: true,
              onTap: () => onNotificationTap(
                  notif["zone"] as Map<String, dynamic>),
              leading: Icon(DangerStyle.icon(danger), color: color, size: 22),
              title: Text(
                message,
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 12,
                  fontFamily: _fontFamily,
                ),
              ),
              subtitle: Text(
                "Tap to view on the map",
                style: TextStyle(
                  color: color.withOpacity(0.9),
                  fontSize: 9.5,
                  fontFamily: _fontFamily,
                ),
              ),
            ),
          ),
        );
      },
    );
  }
}
