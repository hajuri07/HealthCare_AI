import 'package:flutter/material.dart';
import '../theme.dart';
import 'skin_analysis_screen.dart';
import 'mock_feature_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Row(
          children: [
            const Text("Dermalyze", style: TextStyle(fontWeight: FontWeight.w700, fontSize: 20)),
            const SizedBox(width: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
              decoration: BoxDecoration(
                color: AppTheme.primary.withOpacity(0.1),
                borderRadius: BorderRadius.circular(20),
              ),
              child: const Text("BETA", style: TextStyle(fontSize: 11, color: AppTheme.primary, fontWeight: FontWeight.w600)),
            ),
          ],
        ),
      ),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: GridView.count(
          crossAxisCount: 2,
          mainAxisSpacing: 16,
          crossAxisSpacing: 16,
          children: [
            _FeatureCard(
              icon: Icons.camera_alt_outlined,
              label: "Skin Analysis",
              onTap: () => Navigator.push(context,
                MaterialPageRoute(builder: (_) => const SkinAnalysisScreen())),
            ),
            _FeatureCard(
              icon: Icons.monitor_heart_outlined,
              label: "MRI Triage",
              onTap: () => Navigator.push(context,
                MaterialPageRoute(builder: (_) => const MockFeatureScreen(
                  title: "MRI Triage",
                  description: "Upload MRI scans for AI-assisted preliminary triage.",
                  mockResult: "Preliminary result: No significant abnormality detected. Recommend routine follow-up.",
                ))),
            ),
          ],
        ),
      ),
    );
  }
}

class _FeatureCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;
  const _FeatureCard({required this.icon, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return Card(
      child: InkWell(
        borderRadius: BorderRadius.circular(16),
        onTap: onTap,
        child: Padding(
          padding: const EdgeInsets.all(20),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, size: 36, color: AppTheme.primary),
              const SizedBox(height: 12),
              Text(label, textAlign: TextAlign.center,
                style: const TextStyle(fontWeight: FontWeight.w500)),
            ],
          ),
        ),
      ),
    );
  }
}