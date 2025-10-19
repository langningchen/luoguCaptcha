<?php

require 'vendor/autoload.php';

use Gregwar\Captcha\CaptchaBuilder;

if ($argc == 1) {
  $builder = new CaptchaBuilder(4);
  $builder->build($width = 90, $height = 35);
  print($builder->getPhrase());
  $builder->save('captcha.jpg');
} else {
  $tot = intval($argv[1]);
  mt_srand(microtime(true) + intval($argv[2]) + getmypid());
  $ostream = fopen("php://stdout", "wb");
  for ($i = 0; $i < $tot; ++$i) {
    $builder = new CaptchaBuilder(4);
    $builder->build($width = 90, $height = 35);
    $phrase = $builder->getPhrase();
    $img = $builder->get();
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
