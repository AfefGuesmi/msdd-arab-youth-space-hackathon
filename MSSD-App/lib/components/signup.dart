import 'package:flutter/material.dart';
import './auth_service.dart';
import './map.dart';

class SignUpPage extends StatefulWidget {
  const SignUpPage({super.key});

  @override
  State<SignUpPage> createState() => _SignUpPageState();
}

class _SignUpPageState extends State<SignUpPage> {
  static const String _fontFamily = 'Times New Roman';
  // The exact same two colours used in the sidebar gradient — so this
  // page's blue genuinely matches the sidebars instead of being a
  // separate, darker shade.
  static const Color primaryBlue = Color(0xFF1A1F33);
  static const Color darkNavy = Color(0xFF0A0E1A);

  static const int _minPasswordLength = 6;
  static const int _minUsernameLength = 3;

  final TextEditingController _usernameController = TextEditingController();
  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();
  final TextEditingController _confirmController = TextEditingController();

  bool _isLoading = false;
  bool _obscurePassword = true;
  bool _obscureConfirm = true;

  @override
  void dispose() {
    _usernameController.dispose();
    _emailController.dispose();
    _passwordController.dispose();
    _confirmController.dispose();
    super.dispose();
  }

  bool _isValidEmail(String email) {
    return RegExp(r'^[^@\s]+@[^@\s]+\.[^@\s]+$').hasMatch(email);
  }

  // Checks everything that can be checked on the phone first, then asks the
  // database whether the email / username are still free, and only then
  // saves the account.
  Future<void> _handleSignUp() async {
    FocusScope.of(context).unfocus();

    final String username = _usernameController.text.trim();
    final String email = _emailController.text.trim();
    final String password = _passwordController.text;
    final String confirm = _confirmController.text;

    if (username.isEmpty ||
        email.isEmpty ||
        password.isEmpty ||
        confirm.isEmpty) {
      _showMessage("Please fill in every field.");
      return;
    }
    if (username.length < _minUsernameLength) {
      _showMessage("Username must be at least $_minUsernameLength characters.");
      return;
    }
    if (!_isValidEmail(email)) {
      _showMessage("Please enter a valid email address.");
      return;
    }
    if (password.length < _minPasswordLength) {
      _showMessage("Password must be at least $_minPasswordLength characters.");
      return;
    }
    if (password != confirm) {
      _showMessage("Passwords don't match.");
      return;
    }

    setState(() => _isLoading = true);

    try {
      final AuthResult result = await AuthService.instance.signUp(
        username: username,
        email: email,
        password: password,
      );
      if (!mounted) return;

      switch (result.status) {
        case AuthStatus.success:
          // The new account is already signed in (see AuthService.signUp),
          // so go straight to the map — no sign-in page in between.
          Navigator.pushAndRemoveUntil(
            context,
            MaterialPageRoute(builder: (context) => const MapPage()),
            (route) => false,
          );
          break;
        case AuthStatus.emailTaken:
          _showMessage(
            "An account with this email already exists.",
            actionLabel: "Sign In",
            onAction: () => Navigator.pushNamed(context, '/signin',
                arguments: email),
          );
          break;
        case AuthStatus.usernameTaken:
          _showMessage("This username is already taken. Try another one.");
          break;
        default:
          _showMessage("Something went wrong. Please try again.");
      }
    } catch (_) {
      if (!mounted) return;
      _showMessage("Something went wrong. Please try again.");
    } finally {
      if (mounted) setState(() => _isLoading = false);
    }
  }

  void _showMessage(
    String message, {
    bool success = false,
    String? actionLabel,
    VoidCallback? onAction,
  }) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          backgroundColor: success
              ? Colors.green.shade800.withOpacity(0.95)
              : Colors.red.shade900.withOpacity(0.95),
          content: Text(
            message,
            style: const TextStyle(fontFamily: _fontFamily, fontSize: 13),
          ),
          action: actionLabel == null
              ? null
              : SnackBarAction(
                  label: actionLabel,
                  textColor: Colors.white,
                  onPressed: onAction ?? () {},
                ),
        ),
      );
  }

  // The show/hide eye shown inside a password field.
  Widget _eyeButton(bool obscured, VoidCallback onPressed) {
    return IconButton(
      tooltip: obscured ? "Show password" : "Hide password",
      icon: Icon(
        obscured ? Icons.visibility_outlined : Icons.visibility_off_outlined,
        color: Colors.white70,
        size: 20,
      ),
      onPressed: onPressed,
    );
  }

  // Shared look for all four text fields.
  InputDecoration _fieldDecoration(String hint, IconData icon,
      {Widget? suffixIcon}) {
    return InputDecoration(
      hintText: hint,
      hintStyle: const TextStyle(
        color: Colors.white70,
        fontSize: 13,
        fontFamily: _fontFamily,
      ),
      filled: true,
      fillColor: Colors.black.withOpacity(0.18),
      border: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide.none,
      ),
      enabledBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: BorderSide(color: Colors.white.withOpacity(0.18)),
      ),
      focusedBorder: OutlineInputBorder(
        borderRadius: BorderRadius.circular(12),
        borderSide: const BorderSide(color: Colors.white, width: 1.4),
      ),
      prefixIcon: Icon(icon, color: Colors.white70, size: 20),
      suffixIcon: suffixIcon,
    );
  }

  @override
  Widget build(BuildContext context) {
    const TextStyle fieldStyle = TextStyle(
      color: Colors.white,
      fontSize: 13,
      fontFamily: _fontFamily,
    );

    return Scaffold(
      // The page looks fixed (no scrolling) while everything fits. When the
      // keyboard opens and the room shrinks, the content scrolls instead of
      // overflowing at the bottom.
      resizeToAvoidBottomInset: true,
      body: Container(
        decoration: BoxDecoration(
          // Just the two sidebar colours in a smooth two-stop blend,
          // exactly like the sidebar panels.
          gradient: LinearGradient(
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              primaryBlue,
              darkNavy,
            ],
          ),
        ),
        child: SafeArea(
          child: LayoutBuilder(
            builder: (context, constraints) => SingleChildScrollView(
              keyboardDismissBehavior: ScrollViewKeyboardDismissBehavior.onDrag,
              padding: const EdgeInsets.all(24),
              // At least as tall as the screen, so the sign-in link stays at
              // the bottom; taller than that only when the keyboard is up.
              child: ConstrainedBox(
                constraints: BoxConstraints(
                  minHeight: constraints.maxHeight - 48,
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                IconButton(
                  icon: const Icon(Icons.arrow_back, color: Colors.white),
                  onPressed: () {
                    Navigator.pop(context);
                  },
                ),
                const SizedBox(height: 4),
                // Same short accent line used on the home page, instead
                // of a circular emblem icon.
                Container(
                  width: 42,
                  height: 3,
                  decoration: BoxDecoration(
                    color: primaryBlue,
                    borderRadius: BorderRadius.circular(2),
                  ),
                ),
                const SizedBox(height: 12),
                const Text(
                  "Create an account",
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    fontFamily: _fontFamily,
                    shadows: [
                      Shadow(offset: Offset(0, 1), blurRadius: 4, color: Colors.black45),
                    ],
                  ),
                ),
                const SizedBox(height: 5),
                const Text(
                  "Join us to help track ocean debris",
                  style: TextStyle(
                    color: Colors.white70,
                    fontSize: 12,
                    fontFamily: _fontFamily,
                    shadows: [
                      Shadow(offset: Offset(0, 1), blurRadius: 3, color: Colors.black38),
                    ],
                  ),
                ),
                const SizedBox(height: 18),
                TextField(
                  controller: _usernameController,
                  textInputAction: TextInputAction.next,
                  autocorrect: false,
                  style: fieldStyle,
                  decoration: _fieldDecoration("Username", Icons.person_outline),
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: _emailController,
                  keyboardType: TextInputType.emailAddress,
                  textInputAction: TextInputAction.next,
                  autocorrect: false,
                  style: fieldStyle,
                  decoration: _fieldDecoration("Email", Icons.email_outlined),
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: _passwordController,
                  obscureText: _obscurePassword,
                  textInputAction: TextInputAction.next,
                  style: fieldStyle,
                  decoration: _fieldDecoration(
                    "Password",
                    Icons.lock_outline,
                    suffixIcon: _eyeButton(
                      _obscurePassword,
                      () => setState(() => _obscurePassword = !_obscurePassword),
                    ),
                  ),
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: _confirmController,
                  obscureText: _obscureConfirm,
                  textInputAction: TextInputAction.done,
                  onSubmitted: (_) {
                    if (!_isLoading) _handleSignUp();
                  },
                  style: fieldStyle,
                  decoration: _fieldDecoration(
                    "Confirm password",
                    Icons.lock_outline,
                    suffixIcon: _eyeButton(
                      _obscureConfirm,
                      () => setState(() => _obscureConfirm = !_obscureConfirm),
                    ),
                  ),
                ),
                const SizedBox(height: 20),
                ElevatedButton(
                  style: ElevatedButton.styleFrom(
                    backgroundColor: primaryBlue,
                    foregroundColor: Colors.white,
                    elevation: 6,
                    shadowColor: primaryBlue.withOpacity(0.5),
                    minimumSize: const Size(double.infinity, 52),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                  ),
                  onPressed: _isLoading ? null : _handleSignUp,
                  child: _isLoading
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2.4,
                            valueColor: AlwaysStoppedAnimation(Colors.white),
                          ),
                        )
                      : const Text(
                          "Sign Up",
                          style: TextStyle(fontSize: 15, fontFamily: _fontFamily),
                        ),
                ),
                      ],
                    ),
                    Column(
                      children: [
                const SizedBox(height: 16),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      "Already have an account? ",
                      style: TextStyle(
                        color: Colors.white70,
                        fontSize: 12,
                        fontFamily: _fontFamily,
                        shadows: [
                          Shadow(offset: Offset(0, 1), blurRadius: 3, color: Colors.black38),
                        ],
                      ),
                    ),
                    GestureDetector(
                      onTap: () {
                        Navigator.pushNamed(context, '/signin');
                      },
                      child: const Text(
                        "Sign In",
                        style: TextStyle(
                          color: Colors.white,
                          decoration: TextDecoration.underline,
                          fontWeight: FontWeight.bold,
                          fontSize: 12,
                          fontFamily: _fontFamily,
                          shadows: [
                            Shadow(offset: Offset(0, 1), blurRadius: 3, color: Colors.black38),
                          ],
                        ),
                      ),
                    ),
                  ],
                ),
                // Bottom margin under the closing line so the page
                // doesn't feel cramped against the screen edge.
                const SizedBox(height: 26),
                      ],
                    ),
                  ],
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}
