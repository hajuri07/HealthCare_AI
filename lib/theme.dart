import 'package:flutter/material.dart';

class AppTheme {
  static const Color background = Color(0xFFFAFAF8);
  static const Color primary = Color(0xFF2D6A5C);   // muted sage green, medical-calm not neon-AI
  static const Color surface = Color(0xFFFFFFFF);
  static const Color textDark = Color(0xFF2B2B2B);
  static const Color textMuted = Color(0xFF7A7A7A);

  static ThemeData get theme => ThemeData(
    useMaterial3: true,
    scaffoldBackgroundColor: background,
    colorScheme: ColorScheme.fromSeed(
      seedColor: primary,
      brightness: Brightness.light,
    ),
    appBarTheme: const AppBarTheme(
      backgroundColor: background,
      foregroundColor: textDark,
      elevation: 0,
      centerTitle: false,
    ),
    cardTheme: CardThemeData(
      elevation: 0,
      color: surface,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(color: Colors.grey.shade200),
      ),
    ),
    elevatedButtonTheme: ElevatedButtonThemeData(
      style: ElevatedButton.styleFrom(
        backgroundColor: primary,
        foregroundColor: Colors.white,
        elevation: 0,
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 14),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
      ),
    ),
    textTheme: const TextTheme(
      headlineSmall: TextStyle(fontWeight: FontWeight.w600, color: textDark),
      bodyMedium: TextStyle(color: textDark),
      bodySmall: TextStyle(color: textMuted),
    ),
  );
}