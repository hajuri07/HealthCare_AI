import 'dart:convert';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:flutter/foundation.dart' show kIsWeb;

class ApiService {
  static String get baseUrl {
    if (kIsWeb) return "http://localhost:8000";
    return "http://10.0.2.2:8000"; // Android emulator
  }

  static Future<Map<String, dynamic>> predict(XFile imageFile) async {
    final request = http.MultipartRequest('POST', Uri.parse('$baseUrl/predict'));
    final bytes = await imageFile.readAsBytes();
    request.files.add(http.MultipartFile.fromBytes(
      'file',
      bytes,
      filename: imageFile.name,
    ));

    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);

    if (response.statusCode == 200) {
      return jsonDecode(response.body);
    } else {
      throw Exception("Server error: ${response.statusCode}");
    }
  }
}