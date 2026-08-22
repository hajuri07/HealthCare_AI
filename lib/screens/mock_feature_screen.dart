import 'package:flutter/material.dart';

class MockFeatureScreen extends StatefulWidget {
  final String title;
  final String description;
  final String mockResult;
  const MockFeatureScreen({super.key, required this.title, required this.description, required this.mockResult});

  @override
  State<MockFeatureScreen> createState() => _MockFeatureScreenState();
}

class _MockFeatureScreenState extends State<MockFeatureScreen> {
  bool _loading = false;
  bool _done = false;

  void _run() async {
    setState(() => _loading = true);
    await Future.delayed(const Duration(seconds: 2));
    setState(() { _loading = false; _done = true; });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text(widget.title)),
      body: Padding(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(widget.description, style: const TextStyle(color: Colors.grey)),
            const SizedBox(height: 24),
            ElevatedButton.icon(
              onPressed: _run,
              icon: const Icon(Icons.upload_file_outlined),
              label: const Text("Upload & Analyze"),
            ),
            const SizedBox(height: 24),
            if (_loading) const Center(child: CircularProgressIndicator()),
            if (_done) Card(
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: Text(widget.mockResult),
              ),
            ),
          ],
        ),
      ),
    );
  }
}