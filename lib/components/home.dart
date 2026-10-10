import 'package:flutter/material.dart';

class HomePage extends StatelessWidget {
  const HomePage({super.key});

  static const String _fontFamily = 'Times New Roman';
  static const Color primaryBlue = Color(0xFF005B7D);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        width: double.infinity,
        height: double.infinity,
        decoration: const BoxDecoration(
          image: DecorationImage(
            image: AssetImage("assets/images/1.png"),
            fit: BoxFit.cover,
            alignment: Alignment.center,
          ),
        ),
        child: Container(
          // Subtle gradient overlay tinted with the same dark navy blue
          // used on the sign-in page, instead of a flat black scrim —
          // just enough to tie the two screens together while keeping
          // the photo readable underneath.
          decoration: BoxDecoration(
            gradient: LinearGradient(
              begin: Alignment.topCenter,
              end: Alignment.bottomCenter,
              colors: [
                const Color(0xFF0A0E1A).withOpacity(1),
                const Color(0xFF0A0E1A).withOpacity(0.8),
                const Color(0xFF0A0E1A).withOpacity(0.9),
              ],
              stops: const [0.0, 0.55, 1.0],
            ),
          ),
          child: SafeArea(
            // StackFit.expand keeps the content filling the whole screen, so
            // the buttons stay pinned to the bottom as before.
            child: Stack(
              fit: StackFit.expand,
              children: [
            Padding(
              padding: const EdgeInsets.fromLTRB(24, 20, 24, 20),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.end,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // Small decorative accent line above the wordmark.
                  Container(
                    width: 42,
                    height: 3,
                    decoration: BoxDecoration(
                      color: primaryBlue,
                      borderRadius: BorderRadius.circular(2),
                    ),
                  ),
                  const SizedBox(height: 14),
                  const Text(
                    "MSDD",
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 30,
                      fontWeight: FontWeight.bold,
                      fontFamily: _fontFamily,
                      letterSpacing: 1.2,
                    ),
                  ),
                  const SizedBox(height: 8),
                  const Text(
                    "For Clear Seas and Safe Coastal Life.",
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 12,
                      fontFamily: _fontFamily,
                      fontStyle: FontStyle.italic,
                    ),
                  ),
                  const SizedBox(height: 28),
                  ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: Colors.black,
                      elevation: 4,
                      shadowColor: Colors.black54,
                      minimumSize: const Size(double.infinity, 46),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    onPressed: () {
                      Navigator.pushNamed(context, '/signin');
                    },
                    child: const Text(
                      "Log in",
                      style: TextStyle(
                        fontFamily: _fontFamily,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  const SizedBox(height: 12),
                  OutlinedButton(
                    style: OutlinedButton.styleFrom(
                      side: const BorderSide(color: Colors.white, width: 1.2),
                      foregroundColor: Colors.white,
                      minimumSize: const Size(double.infinity, 46),
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(10),
                      ),
                    ),
                    onPressed: () {
                      Navigator.pushNamed(context, '/signup');
                    },
                    child: const Text(
                      "Register",
                      style: TextStyle(
                        fontFamily: _fontFamily,
                        fontSize: 14,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                  ),
                  const SizedBox(height: 22),
                  // Thin divider gives the footer a finished, deliberate feel.
                  Center(
                    child: Container(
                      width: 60,
                      height: 1,
                      color: Colors.white24,
                    ),
                  ),
                  const SizedBox(height: 10),
                  const Center(
                    child: Text(
                      "Copyright © all rights reserved.",
                      style: TextStyle(
                        color: Colors.white60,
                        fontSize: 9,
                        fontFamily: _fontFamily,
                      ),
                    ),
                  ),
                  const SizedBox(height: 8),
                ],
              ),
            ),
            // "About the app" — a small round info button in the top-right
            // corner that opens the description popup.
            Positioned(
              top: 8,
              right: 12,
              child: Tooltip(
                message: "About the app",
                child: Material(
                  color: Colors.white.withOpacity(0.12),
                  shape: CircleBorder(
                    side: BorderSide(color: Colors.white.withOpacity(0.35)),
                  ),
                  child: InkWell(
                    customBorder: const CircleBorder(),
                    onTap: () => _showAbout(context),
                    child: const Padding(
                      padding: EdgeInsets.all(10),
                      child: Icon(Icons.info_outline,
                          color: Colors.white, size: 20),
                    ),
                  ),
                ),
              ),
            ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  // --- "About the app" popup -------------------------------------------

  static const Color _navy = Color(0xFF0A0E1A);
  static const Color _navy2 = Color(0xFF1A1F33);

  // (icon, title, description) for each feature listed in the popup.
  static const List<List<Object>> _features = [
    [
      Icons.map_outlined,
      "Interactive map",
      "A dark world map showing the zones where marine debris was detected "
          "in satellite imagery. Icons appear as you zoom in and follow the "
          "area you move to.",
    ],
    [
      Icons.traffic_outlined,
      "Danger levels",
      "Every zone is colour-coded: red for alerts (high danger), orange for "
          "warnings (medium) and yellow for notices (low).",
    ],
    [
      Icons.location_on_outlined,
      "Detected zones",
      "Browse every zone in a list. Selecting one zooms the map to it and "
          "shows its debris type, location, recommended action and "
          "qualitative result image.",
    ],
    [
      Icons.description_outlined,
      "Reports",
      "Download a PDF report for a single zone or a global report covering "
          "all zones, with the zone details and its image.",
    ],
    [
      Icons.notifications_outlined,
      "Notifications",
      "One notification per detected area, by severity. Tap one to jump "
          "straight to that area on the map.",
    ],
    [
      Icons.smart_toy_outlined,
      "Assistant",
      "An assistant to answer your questions about the app and the "
          "detections (coming soon).",
    ],
    [
      Icons.lock_outline,
      "Your account",
      "Create an account and sign in to reach the map, and sign out at "
          "any time from the menu.",
    ],
  ];

  void _showAbout(BuildContext context) {
    showDialog<void>(
      context: context,
      barrierColor: Colors.black.withOpacity(0.7),
      builder: (dialogContext) {
        final Size size = MediaQuery.of(dialogContext).size;
        return Dialog(
          backgroundColor: Colors.transparent,
          elevation: 0,
          insetPadding:
              const EdgeInsets.symmetric(horizontal: 20, vertical: 32),
          child: ConstrainedBox(
            constraints: BoxConstraints(
              maxWidth: 520,
              maxHeight: size.height * 0.85,
            ),
            child: Container(
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [_navy2, _navy],
                ),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: Colors.white.withOpacity(0.14)),
                boxShadow: const [
                  BoxShadow(
                    color: Colors.black54,
                    blurRadius: 30,
                    offset: Offset(0, 12),
                  ),
                ],
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Padding(
                    padding: const EdgeInsets.fromLTRB(22, 18, 8, 0),
                    child: Row(
                      children: [
                        Container(
                          width: 34,
                          height: 3,
                          decoration: BoxDecoration(
                            color: Colors.white54,
                            borderRadius: BorderRadius.circular(2),
                          ),
                        ),
                        const Spacer(),
                        IconButton(
                          icon: const Icon(Icons.close,
                              color: Colors.white70, size: 20),
                          onPressed: () => Navigator.pop(dialogContext),
                        ),
                      ],
                    ),
                  ),
                  const Padding(
                    padding: EdgeInsets.fromLTRB(22, 0, 22, 4),
                    child: Text(
                      "About MSDD",
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        fontFamily: _fontFamily,
                        letterSpacing: 0.6,
                      ),
                    ),
                  ),
                  Flexible(
                    child: SingleChildScrollView(
                      padding: const EdgeInsets.fromLTRB(22, 8, 22, 8),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text(
                            "MSDD is an ocean debris detection app. It brings "
                            "together the debris detected in satellite imagery "
                            "and presents it on one map, so that polluted "
                            "areas can be located, assessed by danger level "
                            "and followed up with a recommended action.",
                            style: TextStyle(
                              color: Colors.white70,
                              fontSize: 13,
                              height: 1.5,
                              fontFamily: _fontFamily,
                            ),
                          ),
                          const SizedBox(height: 18),
                          const Text(
                            "WHAT YOU CAN DO",
                            style: TextStyle(
                              color: Colors.white38,
                              fontSize: 10.5,
                              letterSpacing: 1.3,
                              fontWeight: FontWeight.w600,
                              fontFamily: _fontFamily,
                            ),
                          ),
                          const SizedBox(height: 10),
                          for (final f in _features)
                            Padding(
                              padding: const EdgeInsets.only(bottom: 14),
                              child: Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Container(
                                    width: 34,
                                    height: 34,
                                    decoration: BoxDecoration(
                                      shape: BoxShape.circle,
                                      color: Colors.white.withOpacity(0.08),
                                      border: Border.all(
                                          color:
                                              Colors.white.withOpacity(0.2)),
                                    ),
                                    child: Icon(f[0] as IconData,
                                        color: Colors.white, size: 17),
                                  ),
                                  const SizedBox(width: 12),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment:
                                          CrossAxisAlignment.start,
                                      children: [
                                        Text(
                                          f[1] as String,
                                          style: const TextStyle(
                                            color: Colors.white,
                                            fontSize: 13.5,
                                            fontWeight: FontWeight.bold,
                                            fontFamily: _fontFamily,
                                          ),
                                        ),
                                        const SizedBox(height: 2),
                                        Text(
                                          f[2] as String,
                                          style: const TextStyle(
                                            color: Colors.white70,
                                            fontSize: 12,
                                            height: 1.45,
                                            fontFamily: _fontFamily,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                        ],
                      ),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.fromLTRB(22, 4, 22, 20),
                    child: SizedBox(
                      width: double.infinity,
                      child: ElevatedButton(
                        style: ElevatedButton.styleFrom(
                          backgroundColor: Colors.white,
                          foregroundColor: _navy,
                          elevation: 0,
                          padding: const EdgeInsets.symmetric(vertical: 13),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(10),
                          ),
                        ),
                        onPressed: () => Navigator.pop(dialogContext),
                        child: const Text(
                          "Got it",
                          style: TextStyle(
                            fontFamily: _fontFamily,
                            fontSize: 14,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}