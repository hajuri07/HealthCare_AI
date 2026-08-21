import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import '../services/api_service.dart';

class SkinAnalysisScreen extends StatefulWidget {
  const SkinAnalysisScreen({super.key});
  @override
  State<SkinAnalysisScreen> createState() => _SkinAnalysisScreenState();
}

class _SkinAnalysisScreenState extends State<SkinAnalysisScreen> {
  File? _image;
  bool _loading = false;
  Map<String, dynamic>? _result;

  Future<void> _pickImage(ImageSource source) async {
    final picked = await ImagePicker().pickImage(source: source);
    if (picked == null) return;
    setState(() { _image = File(picked.path); _result = null; });
    _predict();
  }

  Future<void> _predict() async {
    setState(() => _loading = true);
    try {
      final result = await ApiService.predict(_image!);
      setState(() => _result = result);
    } catch (e) {
      setState(() => _result = {"error": e.toString()});
    } finally {
      setState(() => _loading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Skin Analysis")),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          children: [
            if (_image != null)
              ClipRRect(borderRadius: BorderRadius.circular(12),
                child: Image.file(_image!, height: 220, fit: BoxFit.cover)),
            const SizedBox(height: 16),
            Row(mainAxisAlignment: MainAxisAlignment.center, children: [
              ElevatedButton.icon(onPressed: () => _pickImage(ImageSource.camera),
                icon: const Icon(Icons.camera_alt), label: const Text("Camera")),
              const SizedBox(width: 12),
              ElevatedButton.icon(onPressed: () => _pickImage(ImageSource.gallery),
                icon: const Icon(Icons.photo_library), label: const Text("Gallery")),
            ]),
            const SizedBox(height: 24),
            if (_loading) const CircularProgressIndicator(),
            if (_result != null && !_result!.containsKey("error")) ...[
              Text("Prediction: ${_result!['prediction']}",
                style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold)),
              Text("Confidence: ${(_result!['confidence'] * 100).toStringAsFixed(1)}%"),
              const SizedBox(height: 12),
              Text(_result!['ai_summary'], style: const TextStyle(fontSize: 15)),
              const SizedBox(height: 16),
              if (_result!['gradcam_image'] != null) ...[
                const Text("Model attention (Grad-CAM):", style: TextStyle(fontWeight: FontWeight.bold)),
                const SizedBox(height: 8),
                ClipRRect(borderRadius: BorderRadius.circular(12),
                  child: Image.memory(base64Decode(_result!['gradcam_image']), height: 220)),
              ],
              const SizedBox(height: 12),
              Text(_result!['disclaimer'],
                style: const TextStyle(fontSize: 12, fontStyle: FontStyle.italic, color: Colors.grey)),
            ],
            if (_result != null && _result!.containsKey("error"))
              Text(_result!["error"], style: const TextStyle(color: Colors.red)),
          ],
        ),
      ),
    );
  }
}