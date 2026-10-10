import 'package:flutter/material.dart';
import './components/home.dart';
import './components/signin.dart';
import './components/signup.dart';

void main() {
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
