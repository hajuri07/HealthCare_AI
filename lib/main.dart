import 'package:flutter/material.dart';
import 'theme.dart';
import 'screens/home_screen.dart';

void main() => runApp(const DermalyzeApp());

class DermalyzeApp extends StatelessWidget {
  const DermalyzeApp({super.key});
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Dermalyze',
      theme: AppTheme.theme,
      home: const HomeScreen(),
    );
  }
}