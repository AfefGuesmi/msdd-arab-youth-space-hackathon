import 'package:path/path.dart';
import 'package:sqflite/sqflite.dart';

/// A signed-in (or just registered) person. The password is never part of
/// this object — only the database layer ever sees the hash.
class AppUser {
  final int id;
  final String username;
  final String email;

  const AppUser({
    required this.id,
    required this.username,
    required this.email,
  });

  factory AppUser.fromMap(Map<String, Object?> map) {
    return AppUser(
      id: map['id'] as int,
      username: map['username'] as String,
      email: map['email'] as String,
    );
  }
}

/// The local SQLite database that stores the app's accounts.
///
/// One table, `users`. Username and email are both UNIQUE and compared
/// case-insensitively (COLLATE NOCASE), so "Dorra@Mail.com" and
/// "dorra@mail.com" are the same account and the database itself refuses
/// duplicates even if the app-level check were ever bypassed.
class UserDatabase {
  UserDatabase._();
  static final UserDatabase instance = UserDatabase._();

  static const String _dbName = 'msdd_users.db';
  static const int _dbVersion = 1;

  // Caching the *Future* (not the opened Database) means two screens asking
  // for the database at the same moment still open it only once.
  Future<Database>? _opening;

  Future<Database> get _database => _opening ??= _open();

  Future<Database> _open() async {
    final String path = join(await getDatabasesPath(), _dbName);
    return openDatabase(
      path,
      version: _dbVersion,
      onCreate: (Database db, int version) async {
        await db.execute('''
          CREATE TABLE users (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
            email         TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            salt          TEXT NOT NULL,
            created_at    TEXT NOT NULL
          )
        ''');
      },
    );
  }

  Future<Map<String, Object?>?> findByEmail(String email) async {
    final Database db = await _database;
    final rows = await db.query(
      'users',
      where: 'email = ?',
      whereArgs: [email],
      limit: 1,
    );
    return rows.isEmpty ? null : rows.first;
  }

  Future<Map<String, Object?>?> findByUsername(String username) async {
    final Database db = await _database;
    final rows = await db.query(
      'users',
      where: 'username = ?',
      whereArgs: [username],
      limit: 1,
    );
    return rows.isEmpty ? null : rows.first;
  }

  /// Inserts a new account and returns its id. Throws a [DatabaseException]
  /// (see `isUniqueConstraintError()`) if the username or email is taken.
  Future<int> insertUser({
    required String username,
    required String email,
    required String passwordHash,
    required String salt,
  }) async {
    final Database db = await _database;
    return db.insert('users', {
      'username': username,
      'email': email,
      'password_hash': passwordHash,
      'salt': salt,
      'created_at': DateTime.now().toIso8601String(),
    });
  }
}
