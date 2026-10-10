import 'package:flutter/material.dart';

/// Palette de couleurs centralisée de l'application.
/// Concept : bleu nuit océanique + sarcelle élégante, tons doux pour
/// les niveaux de danger (moins criards que du rouge/vert/jaune purs).
class AppColors {
  AppColors._();

  // Fond
  static const Color background = Color(0xFF0A0F1E);
  static const Color backgroundEnd = Color(0xFF141C30);
  static const List<Color> backgroundGradient = [background, backgroundEnd];

  // Surfaces (cartes, tiroirs, champs)
  static const Color surface = Color(0xFF161F35);
  static const Color surfaceElevated = Color(0xFF1E2A45);
  static const Color surfaceField = Color(0x141FFFFF); // overlay léger

  // Accent principal (sarcelle profonde) et secondaire (bleu ciel doux)
  static const Color primary = Color(0xFF0F8B9E);
  static const Color primaryDark = Color(0xFF0B6B7A);
  static const Color accent = Color(0xFF5FD3D3);

  // Texte
  static const Color textPrimary = Color(0xFFF3F6FA);
  static const Color textSecondary = Color(0xFFA6B0C3);
  static const Color textMuted = Color(0xFF6E7891);

  // Bordures / séparateurs
  static const Color divider = Color(0xFF283252);
  static const Color border = Color(0xFF2C3A5C);

  // Niveaux de danger — tons désaturés, plus élégants
  static const Color dangerLow = Color(0xFF4CAF83);
  static const Color dangerMedium = Color(0xFFD9B24C);
  static const Color dangerHigh = Color(0xFFDB8A4B);
  static const Color dangerCritical = Color(0xFFD9584F);
  static const Color dangerDefault = Color(0xFF5B8DBF);

  static Color dangerColor(String level) {
    switch (level) {
      case "Low":
        return dangerLow;
      case "Medium":
        return dangerMedium;
      case "High":
        return dangerHigh;
      case "Critical":
        return dangerCritical;
      default:
        return dangerDefault;
    }
  }
}

class AppGradients {
  AppGradients._();

  static const LinearGradient background = LinearGradient(
    colors: AppColors.backgroundGradient,
    begin: Alignment.topCenter,
    end: Alignment.bottomCenter,
  );

  static const LinearGradient primaryButton = LinearGradient(
    colors: [AppColors.primary, AppColors.primaryDark],
    begin: Alignment.centerLeft,
    end: Alignment.centerRight,
  );

  static const LinearGradient card = LinearGradient(
    colors: [Color(0xFF1B2540), Color(0xFF141C30)],
    begin: Alignment.topLeft,
    end: Alignment.bottomRight,
  );
}

/// ThemeData global optionnel — à brancher dans MaterialApp(theme: appTheme)
final ThemeData appTheme = ThemeData(
  brightness: Brightness.dark,
  scaffoldBackgroundColor: AppColors.background,
  primaryColor: AppColors.primary,
  fontFamily: 'Inter',
  colorScheme: const ColorScheme.dark(
    primary: AppColors.primary,
    secondary: AppColors.accent,
    surface: AppColors.surface,
  ),
  textTheme: const TextTheme(
    bodyMedium: TextStyle(color: AppColors.textPrimary),
  ),
  dividerColor: AppColors.divider,
  appBarTheme: const AppBarTheme(
    backgroundColor: AppColors.background,
    elevation: 0,
    foregroundColor: AppColors.textPrimary,
  ),
);

/// Style de champ de texte réutilisable pour signin/signup
InputDecoration appInputDecoration({
  required String hint,
  required IconData icon,
}) {
  return InputDecoration(
    hintText: hint,
    hintStyle: const TextStyle(color: AppColors.textMuted, fontSize: 14),
    filled: true,
    fillColor: AppColors.surfaceElevated,
    contentPadding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
    border: OutlineInputBorder(
      borderRadius: BorderRadius.circular(14),
      borderSide: const BorderSide(color: AppColors.border, width: 1),
    ),
    enabledBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(14),
      borderSide: const BorderSide(color: AppColors.border, width: 1),
    ),
    focusedBorder: OutlineInputBorder(
      borderRadius: BorderRadius.circular(14),
      borderSide: const BorderSide(color: AppColors.accent, width: 1.4),
    ),
    prefixIcon: Icon(icon, color: AppColors.textSecondary, size: 20),
  );
}

/// Bouton principal réutilisable (dégradé sarcelle)
class PrimaryButton extends StatelessWidget {
  final String label;
  final VoidCallback onPressed;
  final IconData? icon;

  const PrimaryButton({
    super.key,
    required this.label,
    required this.onPressed,
    this.icon,
  });

  @override
  Widget build(BuildContext context) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        borderRadius: BorderRadius.circular(14),
        onTap: onPressed,
        child: Ink(
          height: 55,
          decoration: BoxDecoration(
            gradient: AppGradients.primaryButton,
            borderRadius: BorderRadius.circular(14),
            boxShadow: [
              BoxShadow(
                color: AppColors.primary.withOpacity(0.35),
                blurRadius: 16,
                offset: const Offset(0, 6),
              ),
            ],
          ),
          child: Center(
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                if (icon != null) ...[
                  Icon(icon, color: Colors.white, size: 20),
                  const SizedBox(width: 8),
                ],
                Text(
                  label,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.w600,
                    letterSpacing: 0.2,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}