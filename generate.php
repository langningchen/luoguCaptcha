<!-- 
Copyright (C) 2025 Langning Chen

This file is part of luoguCaptcha.

luoguCaptcha is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

luoguCaptcha is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with luoguCaptcha.  If not, see <https://www.gnu.org/licenses/>.
-->

<?php

require 'vendor/autoload.php';

use Gregwar\Captcha\CaptchaBuilder;

$builder = new CaptchaBuilder(4);

if ($argc != 2) {
  $builder->build($width = 90, $height = 35);
  print($builder->getPhrase());
  $builder->save('captcha.jpg');
} else {
  $tot = intval($argv[1]);

  $ostream = fopen("php://stdout", "wb");
  for ($i = 0; $i < $tot; ++$i) {
    $builder->build($width = 90, $height = 35);
    $phrase = $builder->getPhrase();
    $img = $builder->inline();
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
