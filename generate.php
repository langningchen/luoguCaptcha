<?php
// error_reporting(0);

function drawLine($image, $width, $height, $tcol = null)
{
	$red = mt_rand(100, 255);
	$green = mt_rand(100, 255);
	$blue = mt_rand(100, 255);
	if ($tcol === null) $tcol = imagecolorallocate($image, $red, $green, $blue);
	if (mt_rand(0, 1)) {
		// Horizontal
		$Xa   = mt_rand(0, $width / 2);
		$Ya   = mt_rand(0, $height);
		$Xb   = mt_rand($width / 2, $width);
		$Yb   = mt_rand(0, $height);
	} else {
		// Vertical
		$Xa   = mt_rand(0, $width);
		$Ya   = mt_rand(0, $height / 2);
		$Xb   = mt_rand(0, $width);
		$Yb   = mt_rand($height / 2, $height);
	}
	imagesetthickness($image, mt_rand(1, 3));
	imageline($image, $Xa, $Ya, $Xb, $Yb, $tcol);
}
/**
 * Apply some post effects
 */
function postEffect($image)
{
	// Negate ?
	if (mt_rand(0, 1) == 0) imagefilter($image, IMG_FILTER_NEGATE);
	// Edge ?
	if (mt_rand(0, 10) == 0) imagefilter($image, IMG_FILTER_EDGEDETECT);
	// Contrast
	imagefilter($image, IMG_FILTER_CONTRAST, mt_rand(-50, 10));
	// Colorize
	if (mt_rand(0, 5) == 0) imagefilter($image, IMG_FILTER_COLORIZE, mt_rand(-80, 50), mt_rand(-80, 50), mt_rand(-80, 50));
}
/**
 * Writes the phrase on the image
 */
function writePhrase($image, $phrase, $font, $width, $height)
{
	// Gets the text size and start position
	$size = $width / 4 - mt_rand(0, 3) - 1;
	$box = \imagettfbbox($size, 0, $font, $phrase);
	$textWidth = $box[2] - $box[0];
	$textHeight = $box[1] - $box[7];
	$x = ($width - $textWidth) / 2;
	$y = ($height - $textHeight) / 2 + $size;
	$textColor = array(mt_rand(0, 150), mt_rand(0, 150), mt_rand(0, 150));
	$col = \imagecolorallocate($image, $textColor[0], $textColor[1], $textColor[2]);
	// Write the letters one by one, with random angle
	for ($i = 0; $i < 4; ++$i) {
		$symbol = mb_substr($phrase, $i, 1);
		$box = \imagettfbbox($size, 0, $font, $symbol);
		$w = $box[2] - $box[0];
		$angle = mt_rand(-8, 8);
		$offset = mt_rand(-5, 5);
		\imagettftext($image, $size, $angle, $x, $y + $offset, $col, $font, $symbol);
		$x += $w;
	}
	return $col;
}
/**
 * Generate the image
 */
function build($width, $height, $phrase)
{
	$font = __DIR__ . '/fonts/captcha' . mt_rand(1, 2) . '.ttf';
	// if background images list is not set, use a color fill as a background
	$image = imagecreatetruecolor($width, $height);
	$bg = imagecolorallocate($image, mt_rand(200, 255), mt_rand(200, 255), mt_rand(200, 255));
	$background = $bg;
	imagefill($image, 0, 0, $bg);
	// Apply effects
	$square = $width * $height;
	drawLine($image, $width, $height);
	// Write CAPTCHA text
	$color = writePhrase($image, $phrase, $font, $width, $height);
	// Apply effects
	$square = $width * $height;
	drawLine($image, $width, $height, $color);
	// Distort the image
	$image = distort($image, $width, $height, $bg);
	// Post effects
	postEffect($image);
	imagejpeg($image, null, 90);
	imagedestroy($image);
}
/**
 * Distorts the image
 */
function distort($image, $width, $height, $bg)
{
	$contents = imagecreatetruecolor($width, $height);
	$X          = mt_rand(0, $width);
	$Y          = mt_rand(0, $height);
	$phase      = mt_rand(0, 10);
	$scale      = 1.1 + mt_rand(0, 10000) / 30000;
	for ($x = 0; $x < $width; $x++) {
		for ($y = 0; $y < $height; $y++) {
			$Vx = $x - $X;
			$Vy = $y - $Y;
			$Vn = sqrt($Vx * $Vx + $Vy * $Vy);
			if ($Vn != 0) {
				$Vn2 = $Vn + 4 * sin($Vn / 30);
				$nX  = $X + ($Vx * $Vn2 / $Vn);
				$nY  = $Y + ($Vy * $Vn2 / $Vn);
			} else {
				$nX = $X;
				$nY = $Y;
			}
			$nY = $nY + $scale * sin($phase + $nX * 0.2);
			$p = interpolate(
				$nX - floor($nX),
				$nY - floor($nY),
				getCol($image, floor($nX), floor($nY), $bg),
				getCol($image, ceil($nX), floor($nY), $bg),
				getCol($image, floor($nX), ceil($nY), $bg),
				getCol($image, ceil($nX), ceil($nY), $bg)
			);
			imagesetpixel($contents, $x, $y, $p);
		}
	}
	return $contents;
}
/**
 * @param $x
 * @param $y
 * @param $nw
 * @param $ne
 * @param $sw
 * @param $se
 *
 * @return int
 */
function interpolate($x, $y, $nw, $ne, $sw, $se)
{
	list($r0, $g0, $b0) = getRGB($nw);
	list($r1, $g1, $b1) = getRGB($ne);
	list($r2, $g2, $b2) = getRGB($sw);
	list($r3, $g3, $b3) = getRGB($se);
	$cx = 1.0 - $x;
	$cy = 1.0 - $y;
	$m0 = $cx * $r0 + $x * $r1;
	$m1 = $cx * $r2 + $x * $r3;
	$r  = (int) ($cy * $m0 + $y * $m1);
	$m0 = $cx * $g0 + $x * $g1;
	$m1 = $cx * $g2 + $x * $g3;
	$g  = (int) ($cy * $m0 + $y * $m1);
	$m0 = $cx * $b0 + $x * $b1;
	$m1 = $cx * $b2 + $x * $b3;
	$b  = (int) ($cy * $m0 + $y * $m1);
	return ($r << 16) | ($g << 8) | $b;
}
/**
 * @param $image
 * @param $x
 * @param $y
 *
 * @return int
 */
function getCol($image, $x, $y, $background)
{
	$L = imagesx($image);
	$H = imagesy($image);
	if ($x < 0 || $x >= $L || $y < 0 || $y >= $H) return $background;
	return imagecolorat($image, $x, $y);
}
/**
 * @param $col
 *
 * @return array
 */
function getRGB($col)
{
	return array((int)($col >> 16) & 0xff, (int)($col >> 8) & 0xff, (int)($col) & 0xff);
}
function build_phrase()
{
	$phrase = '';
	$chars = str_split('abcdefghijklmnpqrstuvwxyz123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ');
	for ($i = 0; $i < 4; $i++) $phrase .= $chars[array_rand($chars)];
	return $phrase;
}
if ($argc != 2) {
	$phrase = build_phrase();
	ob_start();
	build(90, 35, $phrase);
	$img = ob_get_clean();
	$len = strlen($img);
	$out = '';
	$out .= chr(($len & 0xff00) >> 8);
	$out .= chr($len & 0xff);
	echo $phrase . "\n";
	$ostream = fopen("captcha.jpg", "wb");
	fwrite($ostream, $img, $len);
	fflush($ostream);
} else {
	$tot = intval($argv[1]);

	$ostream = fopen("php://stdout", "wb");
	for ($i = 0; $i < $tot; ++$i) {
		$phrase = build_phrase();
		ob_start();
		build(90, 35, $phrase);
		$img = ob_get_clean();
		$len = strlen($img);
		$out = '';
		$out .= chr(($len & 0xff00) >> 8);
		$out .= chr($len & 0xff);
		fwrite($ostream, $out, 2);
		fwrite($ostream, $phrase, 4);
		fwrite($ostream, $img, $len);
		fflush($ostream);
	}
}
