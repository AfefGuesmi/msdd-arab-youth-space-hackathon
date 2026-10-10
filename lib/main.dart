import 'package:flutter/foundation.dart' show kIsWeb;
import 'package:flutter/material.dart';
import 'package:sqflite/sqflite.dart';
import 'package:sqflite_common_ffi_web/sqflite_ffi_web.dart';

import './components/home.dart';
import './components/signin.dart';
import './components/signup.dart';

void main() {
  // In a browser (this is how the Docker version runs) there is no native
  // SQLite, so sqflite is pointed at its web implementation, which stores the
  // database in the browser. On Android / iOS nothing changes.
  if (kIsWeb) {
    databaseFactory = databaseFactoryFfiWeb;
  }
  runApp(const MSDDApp());
}

class MSDDApp extends StatelessWidget {
  const MSDDApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'MSDD',
      theme: ThemeData(
        fontFamily: 'Roboto',
      ),
      initialRoute: '/',
      routes: {
        '/': (context) => const HomePage(),
        '/signin': (context) => const SignInPage(),
        '/signup': (context) => const SignUpPage(),
      },
    );
  }
}
