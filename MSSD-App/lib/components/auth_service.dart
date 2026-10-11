import 'dart:convert';
import 'dart:math';

import 'package:crypto/crypto.dart';
import 'package:sqflite/sqflite.dart' show DatabaseException;

import './user_database.dart';

/// Every possible outcome of a sign-in / sign-up attempt. The pages turn
/// each one into the right message, so no error text lives in this file.
enum AuthStatus {
  success,
  userNotFound, // sign-in: no account with that email
  wrongPassword, // sign-in: account exists, password doesn't match
  emailTaken, // sign-up: an account with that email already exists
  usernameTaken, // sign-up: that username is already used
}

class AuthResult {
  final AuthStatus status;
  final AppUser? user;

  const AuthResult(this.status, [this.user]);

  bool get isSuccess => status == AuthStatus.success;
}

/// Account logic on top of the SQLite [UserDatabase]: registering people,
/// checking their password, and remembering who is signed in.
class AuthService {
  AuthService._();
  static final AuthService instance = AuthService._();

  final UserDatabase _db = UserDatabase.instance;

  AppUser? _currentUser;

  /// The account that signed in successfully, or null if nobody has.
  AppUser? get currentUser => _currentUser;
  bool get isSignedIn => _currentUser != null;

  // ----------------------------------------------------------------
  // Sign up
  // ----------------------------------------------------------------

  /// Creates an account if the email and username are both free, and signs
  /// the new person in right away so the sign-up page can open the map.
  Future<AuthResult> signUp({
    required String username,
    required String email,
    required String password,
  }) async {
    final String cleanUsername = username.trim();
    final String cleanEmail = _normalizeEmail(email);

    if (await _db.findByEmail(cleanEmail) != null) {
      return const AuthResult(AuthStatus.emailTaken);
    }
    if (await _db.findByUsername(cleanUsername) != null) {
      return const AuthResult(AuthStatus.usernameTaken);
    }

    final String salt = _generateSalt();
    try {
      final int id = await _db.insertUser(
        username: cleanUsername,
        email: cleanEmail,
        passwordHash: _hashPassword(password, salt),
        salt: salt,
      );
      final AppUser user =
          AppUser(id: id, username: cleanUsername, email: cleanEmail);
      _currentUser = user;
      return AuthResult(AuthStatus.success, user);
    } on DatabaseException catch (e) {
      // Safety net: the table's UNIQUE constraints catch a duplicate that
      // slipped in between the checks above and the insert.
      if (e.isUniqueConstraintError()) {
        return const AuthResult(AuthStatus.emailTaken);
      }
      rethrow;
    }
  }

  // ----------------------------------------------------------------
  // Sign in
  // ----------------------------------------------------------------

  /// Succeeds only when an account with this email exists AND the password
  /// matches it. Only then is [currentUser] set.
  Future<AuthResult> signIn({
    required String email,
    required String password,
  }) async {
    final Map<String, Object?>? row =
        await _db.findByEmail(_normalizeEmail(email));

    if (row == null) {
      return const AuthResult(AuthStatus.userNotFound);
    }

    final String salt = row['salt'] as String;
    final String storedHash = row['password_hash'] as String;

    if (!_constantTimeEquals(_hashPassword(password, salt), storedHash)) {
      return const AuthResult(AuthStatus.wrongPassword);
    }

    final AppUser user = AppUser.fromMap(row);
    _currentUser = user;
    return AuthResult(AuthStatus.success, user);
  }

  void signOut() {
    _currentUser = null;
  }

  // ----------------------------------------------------------------
  // Helpers
  // ----------------------------------------------------------------

  String _normalizeEmail(String email) => email.trim().toLowerCase();

  // Passwords are never stored. What's stored is a salted, repeatedly
  // hashed digest: the random per-account salt means two people with the
  // same password get different hashes, and the extra rounds make guessing
  // passwords from a stolen database file slower.
  static const int _hashRounds = 10000;

  String _hashPassword(String password, String salt) {
    final List<int> saltBytes = utf8.encode(salt);
    Digest digest = sha256.convert([...saltBytes, ...utf8.encode(password)]);
    for (int i = 0; i < _hashRounds; i++) {
      digest = sha256.convert([...digest.bytes, ...saltBytes]);
    }
    return digest.toString();
  }

  String _generateSalt() {
    final Random random = Random.secure();
    final List<int> bytes = List<int>.generate(16, (_) => random.nextInt(256));
    return base64UrlEncode(bytes);
  }

  // Compares two strings without stopping at the first difference.
  bool _constantTimeEquals(String a, String b) {
    if (a.length != b.length) return false;
    int diff = 0;
    for (int i = 0; i < a.length; i++) {
      diff |= a.codeUnitAt(i) ^ b.codeUnitAt(i);
    }
    return diff == 0;
  }
}
