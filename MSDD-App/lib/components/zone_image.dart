import 'package:flutter/material.dart';

/// Shows the qualitative-result image assigned to a detected zone
/// (assets/qualitative_results/r1.png ... r3.png), with a soft placeholder
/// if the zone has no image or the asset can't be found — so a missing or
/// misspelled file never crashes a sidebar.
class ZoneImage extends StatelessWidget {
  final String? path;
  final double? width;
  final double? height;
  final BoxFit fit;
  final double radius;

  const ZoneImage({
    super.key,
    required this.path,
    this.width,
    this.height,
    this.fit = BoxFit.cover,
    this.radius = 10,
  });

  Widget _placeholder() {
    return Container(
      width: width,
      // A fixed fallback height keeps the placeholder valid inside
      // scrolling lists, where unbounded height would otherwise throw.
      height: height ?? 140,
      alignment: Alignment.center,
      color: Colors.white.withOpacity(0.06),
      child: const Icon(
        Icons.image_not_supported_outlined,
        color: Colors.white38,
        size: 22,
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return ClipRRect(
      borderRadius: BorderRadius.circular(radius),
      child: path == null
          ? _placeholder()
          : Image.asset(
              path!,
              width: width,
              height: height,
              fit: fit,
              errorBuilder: (context, error, stackTrace) => _placeholder(),
            ),
    );
  }
}
