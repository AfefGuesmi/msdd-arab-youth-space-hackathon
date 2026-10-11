import 'package:flutter/material.dart';
import './map.dart';
import './auth_service.dart';

class SignInPage extends StatefulWidget {
  const SignInPage({super.key});

  @override
  State<SignInPage> createState() => _SignInPageState();
}

class _SignInPageState extends State<SignInPage> {
  static const String _fontFamily = 'Times New Roman';
  // The exact same two colours used in the sidebar gradient — so this
  // page's blue genuinely matches the sidebars instead of being a
  // separate, darker shade.
  static const Color primaryBlue = Color(0xFF1A1F33);
  static const Color darkNavy = Color(0xFF0A0E1A);

  final TextEditingController _emailController = TextEditingController();
  final TextEditingController _passwordController = TextEditingController();

  bool _isLoading = false;
  bool _obscurePassword = true;
  bool _prefilled = false;

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    // Coming from a successful sign-up, the new account's email is passed
    // in as the route argument so the person only has to type the password.
    if (!_prefilled) {
      _prefilled = true;
      final Object? args = ModalRoute.of(context)?.settings.arguments;
      if (args is String) {
        _emailController.text = args;
      }
    }
  }

  @override
  void dispose() {
    _emailController.dispose();
    _passwordController.dispose();
    super.dispose();
  }

  // The map is only ever reached from here, and only when the email AND
  // password match a stored account.
  Future<void> _handleSignIn() async {
    FocusScope.of(context).unfocus();

    final String email = _emailController.text.trim();
    final String password = _passwordController.text;

    if (email.isEmpty || password.isEmpty) {
      _showMessage("Please enter your email and password.");
      return;
    }

    setState(() => _isLoading = true);

    try {
      final AuthResult result = await AuthService.instance.signIn(
        email: email,
        password: password,
      );
      if (!mounted) return;

      switch (result.status) {
        case AuthStatus.success:
          Navigator.pushReplacement(
            context,
            MaterialPageRoute(builder: (context) => const MapPage()),
          );
          break;
        case AuthStatus.userNotFound:
          _showMessage(
            "No account found for this email. Please create an account first.",
            actionLabel: "Sign Up",
            onAction: () => Navigator.pushNamed(context, '/signup'),
          );
          break;
        case AuthStatus.wrongPassword:
          _showMessage("Incorrect password. Please try again.");
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
    String? actionLabel,
    VoidCallback? onAction,
  }) {
    ScaffoldMessenger.of(context)
      ..hideCurrentSnackBar()
      ..showSnackBar(
        SnackBar(
          behavior: SnackBarBehavior.floating,
          backgroundColor: Colors.red.shade900.withOpacity(0.95),
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

  // Shared look for both text fields.
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
    return Scaffold(
      // Fixed, non-scrolling layout — the Spacer below absorbs whatever
      // room is left between the form and the bottom line.
      resizeToAvoidBottomInset: true,
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            // Just the two sidebar colours in a smooth two-stop blend,
            // exactly like the sidebar panels.
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
            colors: [
              primaryBlue,
              darkNavy,
            ],
          ),
        ),
        child: SafeArea(
          child: Padding(
            padding: const EdgeInsets.all(24),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                IconButton(
                  icon: const Icon(Icons.arrow_back, color: Colors.white),
                  onPressed: () {
                    Navigator.pop(context);
                  },
                ),
                const SizedBox(height: 6),
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
                const SizedBox(height: 14),
                const Text(
                  "Welcome back!",
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                    fontFamily: _fontFamily,
                    shadows: [
                      Shadow(offset: Offset(0, 1), blurRadius: 4, color: Colors.black45),
                    ],
                  ),
                ),
                const SizedBox(height: 6),
                const Text(
                  "Log in to access the map",
                  style: TextStyle(
                    color: Colors.white70,
                    fontSize: 12,
                    fontFamily: _fontFamily,
                    shadows: [
                      Shadow(offset: Offset(0, 1), blurRadius: 3, color: Colors.black38),
                    ],
                  ),
                ),
                const SizedBox(height: 26),
                TextField(
                  controller: _emailController,
                  keyboardType: TextInputType.emailAddress,
                  textInputAction: TextInputAction.next,
                  autocorrect: false,
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 13,
                    fontFamily: _fontFamily,
                  ),
                  decoration: _fieldDecoration("E-mail", Icons.email_outlined),
                ),
                const SizedBox(height: 14),
                TextField(
                  controller: _passwordController,
                  obscureText: _obscurePassword,
                  textInputAction: TextInputAction.done,
                  onSubmitted: (_) {
                    if (!_isLoading) _handleSignIn();
                  },
                  style: const TextStyle(
                    color: Colors.white,
                    fontSize: 13,
                    fontFamily: _fontFamily,
                  ),
                  decoration: _fieldDecoration(
                    "Password",
                    Icons.lock_outline,
                    suffixIcon: IconButton(
                      tooltip: _obscurePassword
                          ? "Show password"
                          : "Hide password",
                      icon: Icon(
                        _obscurePassword
                            ? Icons.visibility_outlined
                            : Icons.visibility_off_outlined,
                        color: Colors.white70,
                        size: 20,
                      ),
                      onPressed: () =>
                          setState(() => _obscurePassword = !_obscurePassword),
                    ),
                  ),
                ),
                const SizedBox(height: 6),
                Align(
                  alignment: Alignment.centerRight,
                  child: TextButton(
                    onPressed: () {},
                    child: const Text(
                      "Forgot password?",
                      style: TextStyle(
                        color: Colors.white,
                        decoration: TextDecoration.underline,
                        fontWeight: FontWeight.bold,
                        fontFamily: _fontFamily,
                        fontSize: 12,
                        shadows: [
                          Shadow(offset: Offset(0, 1), blurRadius: 3, color: Colors.black38),
                        ],
                      ),
                    ),
                  ),
                ),
                const SizedBox(height: 22),
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
                  onPressed: _isLoading ? null : _handleSignIn,
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
                          "Sign In",
                          style: TextStyle(fontSize: 15, fontFamily: _fontFamily),
                        ),
                ),
                const Spacer(),
                Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    const Text(
                      "Don't have an account? ",
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
                        Navigator.pushNamed(context, '/signup');
                      },
                      child: const Text(
                        "Sign Up",
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
          ),
        ),
      ),
    );
  }
}
