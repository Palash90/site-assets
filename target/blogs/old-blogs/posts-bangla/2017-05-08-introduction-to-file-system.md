---
title: "ফাইল সিস্টেমে হাতেখড়ি"
date: "2017-05-25T23:30:00.000-07:00"
original_url: "http://linux.palash90.in/2017/07/blog-post_24.html"
blog: "লিনাক্স বাংলা গাইড (linux.palash90.in)"
recovered_from: "Atom feed, web.archive.org snapshot 2017-12-03"
---

এর আগের অধ্যায়ে আমরা কিছু অতি সাধারণ কম্যান্ড রান করে দেখেছি সেগুলি কিভাবে কাজ করে, আমরা শেল ব্যবহার করে কিছু সাধারণ গাণিতিক মান জানতে পেরেছি, এইবার আমরা দেখে নেবো লিনাক্স সিস্টেমে ফাইল নিয়ে কিভাবে কাজ করতে পারি।

যারা লিনাক্স ব্যবহার করার আগে Windows বা Mac OS ব্যবহার করেছেন বা Android ফোন ব্যবহার করেছেন, তাদের কাছে ফাইল বিষয়টি নতুন কিছু না কিন্তু যারা এর আগে এই ধরণের কোনো ডিভাইস ব্যবহার করেননি, তাদের মনে প্রশ্ন আসতে পারে, এই ফাইল জিনিসটি কি এবং এটি কি কাজ করে। তাই আমরা এখন খুব সাধারণ ভাষায় ফাইল কি সেটা জেনে নেবার চেষ্টা করবো -
ফাইল হলো byte (বাইট) এর সমন্বয়। তাহলে আপনার প্রশ্ন, এই byte কি। সেটা জানতে গেলে আমাদের একটু গভীরে প্রবেশ করতে হবে। আসলে কম্পিউটারে বিভিন্ন তথ্য আদান প্রদান এবং তথ্য সঞ্চয় করে রাখা হয় বাইনারি ফরম্যাটে। অর্থাৎ যেকোনো তথ্য থাকে ০ এবং ১ আকারে। এই ক্ষুদ্রাতিক্ষুদ্র একক কে বলা হয় bit, এবং ৮ টি bit এর একত্র সমন্বয়কে বলা হয় byte। আসুন, একটা উদাহরণের সাহায্যে ব্যাপারটিকে একটু সরল করে বোঝার চেষ্টা করি -
ধরুন একটি সংখ্যা ১২৮ কম্পিউটারকে প্রদান করতে হবে, আপনি কিবোর্ডে ১২৮ টাইপ করলেও কম্পিউটার কিন্তু এটিকে 1000000 হিসেবে ইনপুট নেবে। সেই ভাবেই যদি ধরুন আপনাকে A প্রদান করতে হয়, তখন আপনি কিবোর্ডে A টাইপ করলেও কম্পিউটারের কাছে সেটি 1000001 হিসেবে ইনপুট নেবে।

যদি আপনার বাইনারি সংখ্যা পদ্ধতি সম্পর্কে কৌতূহল থাকে, আপনি এই ব্লগ পোস্টটি পড়তে পারেন -
[বাইনারি সংখ্যা](http://cpbook.subeen.com/2011/08/binary-number-system.html)

সাধারণত যেকোনো ফাইল এই আকারেই তথ্য সঞ্চয় করে রাখে। সুতরাং বৃহৎ অর্থে আমরা বলতেই পারি যে ফাইল যেকোনো তথ্যকে বাইনারি ফরম্যাটে মেমোরিতে সঞ্চয় করে রাখে এবং প্রয়োজন অনুসারে সেই তথ্য কম্পিউটার হিসেব নিকেশ করার কাজে ব্যবহার করে। সংক্ষিপ্ত ভাবে বলা চলে ফাইল একটি ধারক যেটি বাইনারি তথ্য সঞ্চয় করে কম্পিউটারের হিসেব নিকেশের কাজে সাহায্য করে।

আসলে যেকোনো লিনাক্স ভিত্তিক সিস্টেমে ফাইল খুব গুরুত্বপূর্ণ, কারণ লিনাক্স সিস্টেমে সমস্ত কিছু হয় ফাইল নাহলে প্রসেস (প্রসেস সম্পর্কে আমরা পরে জানবো) এবং লিনাক্স সিস্টেম চলার সময়ে বিভিন্ন ধরণের কনফিগারেশন ফাইল পড়ে প্রয়োজনমতো নিজেকে আমাদের সামনে উপস্থাপন করে। তাই আমরা এই অধ্যায়ে দেখে নেবো কিভাবে আমরা বিভিন্ন ফাইল নিয়ে কাজ করতে পারি।

লিনাক্সে সমস্ত ফাইল বিভিন্ন ডিরেক্টরিতে সাজানো থাকে, দেখে নি কিভাবে। আমরা এখানে একটি ছোট চিত্র তুলে ধরবো -

```
/
├── bin -> usr/bin
├── boot
│   ├── config-3.10.0-327.22.2.el7.x86_64
│   ├── config-4.1.12-37.5.1.el7uek.x86_64
│   ├── grub2
│   │   ├── device.map
│   │   ├── fonts
│   │   │   └── unicode.pf2
│   │   ├── grub.cfg
│   │   ├── grubenv
│   │   ├── i386-pc
│   │   │   ├── acpi.mod
│   │   │   ├── adler32.mod
│   │   │   ├── affs.mod
│   │   │   ├── afs.mod
│   │   │   ├── ahci.mod
│   │   │   ├── all_video.mod
│   │   │   ├── aout.mod
│   │   │   ├── archelp.mod
│   │   │   ├── ata.mod
│   │   │   ├── at_keyboard.mod
│   │   │   ├── backtrace.mod
│   │   │   ├── bfs.mod
│   │   │   ├── biosdisk.mod
│   │   │   ├── bitmap.mod
│   │   │   ├── bitmap_scale.mod
│   │   │   ├── blocklist.mod
│   │   │   ├── blscfg.mod
│   │   │   ├── boot.img
│   │   │   ├── boot.mod
│   │   │   ├── bsd.mod
│   │   │   ├── btrfs.mod
│   │   │   ├── bufio.mod
│   │   │   ├── cat.mod
│   │   │   ├── cbfs.mod
│   │   │   ├── cbls.mod
│   │   │   ├── cbmemc.mod
│   │   │   ├── cbtable.mod
│   │   │   ├── cbtime.mod
│   │   │   ├── chain.mod
│   │   │   ├── cmdline_cat_test.mod
│   │   │   ├── cmosdump.mod
│   │   │   ├── cmostest.mod
│   │   │   ├── cmp.mod
│   │   │   ├── command.lst
│   │   │   ├── configfile.mod
│   │   │   ├── core.img
│   │   │   ├── cpio_be.mod
│   │   │   ├── cpio.mod
│   │   │   ├── cpuid.mod
│   │   │   ├── crc64.mod
│   │   │   ├── cryptodisk.mod
│   │   │   ├── crypto.lst
│   │   │   ├── crypto.mod
│   │   │   ├── cs5536.mod
│   │   │   ├── datehook.mod
│   │   │   ├── date.mod
│   │   │   ├── datetime.mod
│   │   │   ├── diskfilter.mod
│   │   │   ├── disk.mod
│   │   │   ├── div_test.mod
│   │   │   ├── dm_nv.mod
│   │   │   ├── drivemap.mod
│   │   │   ├── echo.mod
│   │   │   ├── efiemu32.o
│   │   │   ├── efiemu64.o
│   │   │   ├── efiemu.mod
│   │   │   ├── ehci.mod
│   │   │   ├── elf.mod
│   │   │   ├── eval.mod
│   │   │   ├── exfat.mod
│   │   │   ├── exfctest.mod
│   │   │   ├── ext2.mod
│   │   │   ├── extcmd.mod
│   │   │   ├── fat.mod
│   │   │   ├── file.mod
│   │   │   ├── font.mod
│   │   │   ├── freedos.mod
│   │   │   ├── fshelp.mod
│   │   │   ├── fs.lst
│   │   │   ├── functional_test.mod
│   │   │   ├── gcry_arcfour.mod
│   │   │   ├── gcry_blowfish.mod
│   │   │   ├── gcry_camellia.mod
│   │   │   ├── gcry_cast5.mod
│   │   │   ├── gcry_crc.mod
│   │   │   ├── gcry_des.mod
│   │   │   ├── gcry_dsa.mod
│   │   │   ├── gcry_idea.mod
│   │   │   ├── gcry_md4.mod
│   │   │   ├── gcry_md5.mod
│   │   │   ├── gcry_rfc2268.mod
│   │   │   ├── gcry_rijndael.mod
│   │   │   ├── gcry_rmd160.mod
│   │   │   ├── gcry_rsa.mod
│   │   │   ├── gcry_seed.mod
│   │   │   ├── gcry_serpent.mod
│   │   │   ├── gcry_sha1.mod
│   │   │   ├── gcry_sha256.mod
│   │   │   ├── gcry_sha512.mod
│   │   │   ├── gcry_tiger.mod
│   │   │   ├── gcry_twofish.mod
│   │   │   ├── gcry_whirlpool.mod
│   │   │   ├── gdb.mod
│   │   │   ├── geli.mod
│   │   │   ├── gettext.mod
│   │   │   ├── gfxmenu.mod
│   │   │   ├── gfxterm_background.mod
│   │   │   ├── gfxterm_menu.mod
│   │   │   ├── gfxterm.mod
│   │   │   ├── gptsync.mod
│   │   │   ├── gzio.mod
│   │   │   ├── halt.mod
│   │   │   ├── hashsum.mod
│   │   │   ├── hdparm.mod
│   │   │   ├── hello.mod
│   │   │   ├── help.mod
│   │   │   ├── hexdump.mod
│   │   │   ├── hfs.mod
│   │   │   ├── hfspluscomp.mod
│   │   │   ├── hfsplus.mod
│   │   │   ├── http.mod
│   │   │   ├── iorw.mod
│   │   │   ├── iso9660.mod
│   │   │   ├── jfs.mod
│   │   │   ├── jpeg.mod
│   │   │   ├── keylayouts.mod
│   │   │   ├── keystatus.mod
│   │   │   ├── ldm.mod
│   │   │   ├── legacycfg.mod
│   │   │   ├── legacy_password_test.mod
│   │   │   ├── linux16.mod
│   │   │   ├── linux.mod
│   │   │   ├── loadenv.mod
│   │   │   ├── loopback.mod
│   │   │   ├── lsacpi.mod
│   │   │   ├── lsapm.mod
│   │   │   ├── lsmmap.mod
│   │   │   ├── ls.mod
│   │   │   ├── lspci.mod
│   │   │   ├── luks.mod
│   │   │   ├── lvm.mod
│   │   │   ├── lzopio.mod
│   │   │   ├── macbless.mod
│   │   │   ├── macho.mod
│   │   │   ├── mda_text.mod
│   │   │   ├── mdraid09_be.mod
│   │   │   ├── mdraid09.mod
│   │   │   ├── mdraid1x.mod
│   │   │   ├── memdisk.mod
│   │   │   ├── memrw.mod
│   │   │   ├── minicmd.mod
│   │   │   ├── minix2_be.mod
│   │   │   ├── minix2.mod
│   │   │   ├── minix3_be.mod
│   │   │   ├── minix3.mod
│   │   │   ├── minix_be.mod
│   │   │   ├── minix.mod
│   │   │   ├── mmap.mod
│   │   │   ├── moddep.lst
│   │   │   ├── modinfo.sh
│   │   │   ├── morse.mod
│   │   │   ├── mpi.mod
│   │   │   ├── msdospart.mod
│   │   │   ├── multiboot2.mod
│   │   │   ├── multiboot.mod
│   │   │   ├── nativedisk.mod
│   │   │   ├── net.mod
│   │   │   ├── newc.mod
│   │   │   ├── nilfs2.mod
│   │   │   ├── normal.mod
│   │   │   ├── ntfscomp.mod
│   │   │   ├── ntfs.mod
│   │   │   ├── ntldr.mod
│   │   │   ├── odc.mod
│   │   │   ├── offsetio.mod
│   │   │   ├── ohci.mod
│   │   │   ├── part_acorn.mod
│   │   │   ├── part_amiga.mod
│   │   │   ├── part_apple.mod
│   │   │   ├── part_bsd.mod
│   │   │   ├── part_dfly.mod
│   │   │   ├── part_dvh.mod
│   │   │   ├── part_gpt.mod
│   │   │   ├── partmap.lst
│   │   │   ├── part_msdos.mod
│   │   │   ├── part_plan.mod
│   │   │   ├── part_sun.mod
│   │   │   ├── part_sunpc.mod
│   │   │   ├── parttool.lst
│   │   │   ├── parttool.mod
│   │   │   ├── password.mod
│   │   │   ├── password_pbkdf2.mod
│   │   │   ├── pata.mod
│   │   │   ├── pbkdf2.mod
│   │   │   ├── pbkdf2_test.mod
│   │   │   ├── pcidump.mod
│   │   │   ├── pci.mod
│   │   │   ├── plan9.mod
│   │   │   ├── play.mod
│   │   │   ├── png.mod
│   │   │   ├── priority_queue.mod
│   │   │   ├── probe.mod
│   │   │   ├── procfs.mod
│   │   │   ├── progress.mod
│   │   │   ├── pxechain.mod
│   │   │   ├── pxe.mod
│   │   │   ├── raid5rec.mod
│   │   │   ├── raid6rec.mod
│   │   │   ├── read.mod
│   │   │   ├── reboot.mod
│   │   │   ├── regexp.mod
│   │   │   ├── reiserfs.mod
│   │   │   ├── relocator.mod
│   │   │   ├── romfs.mod
│   │   │   ├── scsi.mod
│   │   │   ├── search_fs_file.mod
│   │   │   ├── search_fs_uuid.mod
│   │   │   ├── search_label.mod
│   │   │   ├── search.mod
│   │   │   ├── sendkey.mod
│   │   │   ├── serial.mod
│   │   │   ├── setjmp.mod
│   │   │   ├── setjmp_test.mod
│   │   │   ├── setpci.mod
│   │   │   ├── sfs.mod
│   │   │   ├── signature_test.mod
│   │   │   ├── sleep.mod
│   │   │   ├── sleep_test.mod
│   │   │   ├── spkmodem.mod
│   │   │   ├── squash4.mod
│   │   │   ├── syslinuxcfg.mod
│   │   │   ├── tar.mod
│   │   │   ├── terminal.lst
│   │   │   ├── terminal.mod
│   │   │   ├── terminfo.mod
│   │   │   ├── test_blockarg.mod
│   │   │   ├── testload.mod
│   │   │   ├── test.mod
│   │   │   ├── testspeed.mod
│   │   │   ├── tftp.mod
│   │   │   ├── tga.mod
│   │   │   ├── time.mod
│   │   │   ├── trig.mod
│   │   │   ├── tr.mod
│   │   │   ├── truecrypt.mod
│   │   │   ├── true.mod
│   │   │   ├── udf.mod
│   │   │   ├── ufs1_be.mod
│   │   │   ├── ufs1.mod
│   │   │   ├── ufs2.mod
│   │   │   ├── uhci.mod
│   │   │   ├── usb_keyboard.mod
│   │   │   ├── usb.mod
│   │   │   ├── usbms.mod
│   │   │   ├── usbserial_common.mod
│   │   │   ├── usbserial_ftdi.mod
│   │   │   ├── usbserial_pl2303.mod
│   │   │   ├── usbserial_usbdebug.mod
│   │   │   ├── usbtest.mod
│   │   │   ├── vbe.mod
│   │   │   ├── verify.mod
│   │   │   ├── vga.mod
│   │   │   ├── vga_text.mod
│   │   │   ├── video_bochs.mod
│   │   │   ├── video_cirrus.mod
│   │   │   ├── video_colors.mod
│   │   │   ├── video_fb.mod
│   │   │   ├── videoinfo.mod
│   │   │   ├── video.lst
│   │   │   ├── video.mod
│   │   │   ├── videotest_checksum.mod
│   │   │   ├── videotest.mod
│   │   │   ├── xfs.mod
│   │   │   ├── xnu.mod
│   │   │   ├── xnu_uuid.mod
│   │   │   ├── xnu_uuid_test.mod
│   │   │   ├── xzio.mod
│   │   │   ├── zfscrypt.mod
│   │   │   ├── zfsinfo.mod
│   │   │   └── zfs.mod
│   │   ├── locale
│   │   │   ├── ast.mo
│   │   │   ├── ca.mo
│   │   │   ├── da.mo
│   │   │   ├── de_CH.mo
│   │   │   ├── de@hebrew.mo
│   │   │   ├── de.mo
│   │   │   ├── en@arabic.mo
│   │   │   ├── en@cyrillic.mo
│   │   │   ├── en@greek.mo
│   │   │   ├── en@hebrew.mo
│   │   │   ├── en.mo
│   │   │   ├── en@piglatin.mo
│   │   │   ├── en@quot.mo
│   │   │   ├── eo.mo
│   │   │   ├── es.mo
│   │   │   ├── fi.mo
│   │   │   ├── fr.mo
│   │   │   ├── gl.mo
│   │   │   ├── hu.mo
│   │   │   ├── id.mo
│   │   │   ├── it.mo
│   │   │   ├── ja.mo
│   │   │   ├── lt.mo
│   │   │   ├── nl.mo
│   │   │   ├── pa.mo
│   │   │   ├── pl.mo
│   │   │   ├── pt_BR.mo
│   │   │   ├── ru.mo
│   │   │   ├── sl.mo
│   │   │   ├── sv.mo
│   │   │   ├── tr.mo
│   │   │   ├── uk.mo
│   │   │   ├── vi.mo
│   │   │   ├── zh_CN.mo
│   │   │   └── zh_TW.mo
│   │   └── themes
│   │       └── system
│   ├── initramfs-0-rescue-8d9a1061e6b348e68bab0fe7454ab730.img
│   ├── initramfs-3.10.0-327.22.2.el7.x86_64.img
│   ├── initramfs-4.1.12-37.5.1.el7uek.x86_64.img
│   ├── initramfs-4.1.12-37.5.1.el7uek.x86_64kdump.img
│   ├── initrd-plymouth.img
│   ├── symvers-3.10.0-327.22.2.el7.x86_64.gz
│   ├── symvers-4.1.12-37.5.1.el7uek.x86_64.gz
│   ├── System.map-3.10.0-327.22.2.el7.x86_64
│   ├── System.map-4.1.12-37.5.1.el7uek.x86_64
│   ├── vmlinuz-0-rescue-8d9a1061e6b348e68bab0fe7454ab730
│   ├── vmlinuz-3.10.0-327.22.2.el7.x86_64
│   └── vmlinuz-4.1.12-37.5.1.el7uek.x86_64
├── dev
│   ├── autofs
│   ├── block
│   │   ├── 11:0 -> ../sr0
│   │   ├── 8:0 -> ../sda
│   │   ├── 8:1 -> ../sda1
│   │   ├── 8:2 -> ../sda2
│   │   └── 8:3 -> ../sda3
│   ├── bsg
│   │   ├── 1:0:0:0
│   │   └── 2:0:0:0
│   ├── btrfs-control
│   ├── bus
│   │   └── usb
│   │       ├── 001
│   │       │   └── 001
│   │       └── 002
│   │           └── 001
│   ├── cdrom -> sr0
│   ├── char
│   │   ├── 10:144 -> ../nvram
│   │   ├── 10:183 -> ../hwrng
│   │   ├── 10:184 -> ../cpu/microcode
│   │   ├── 10:200 -> ../net/tun
│   │   ├── 10:227 -> ../mcelog
│   │   ├── 10:228 -> ../hpet
│   │   ├── 10:231 -> ../snapshot
│   │   ├── 10:234 -> ../btrfs-control
│   │   ├── 10:235 -> ../autofs
│   │   ├── 10:236 -> ../mapper/control
│   │   ├── 10:57 -> ../vboxuser
│   │   ├── 10:58 -> ../vboxguest
│   │   ├── 10:59 -> ../memory_bandwidth
│   │   ├── 10:60 -> ../network_throughput
│   │   ├── 10:61 -> ../network_latency
│   │   ├── 10:62 -> ../cpu_dma_latency
│   │   ├── 10:63 -> ../vga_arbiter
│   │   ├── 1:1 -> ../mem
│   │   ├── 1:11 -> ../kmsg
│   │   ├── 116:1 -> ../snd/seq
│   │   ├── 116:2 -> ../snd/controlC0
│   │   ├── 116:3 -> ../snd/pcmC0D0p
│   │   ├── 116:33 -> ../snd/timer
│   │   ├── 116:4 -> ../snd/pcmC0D0c
│   │   ├── 116:5 -> ../snd/pcmC0D1c
│   │   ├── 1:3 -> ../null
│   │   ├── 13:32 -> ../input/mouse0
│   │   ├── 13:63 -> ../input/mice
│   │   ├── 13:64 -> ../input/event0
│   │   ├── 13:65 -> ../input/event1
│   │   ├── 13:66 -> ../input/event2
│   │   ├── 13:67 -> ../input/event3
│   │   ├── 13:68 -> ../input/event4
│   │   ├── 13:69 -> ../input/event5
│   │   ├── 13:70 -> ../input/event6
│   │   ├── 1:4 -> ../port
│   │   ├── 1:5 -> ../zero
│   │   ├── 162:0 -> ../raw/rawctl
│   │   ├── 1:7 -> ../full
│   │   ├── 1:8 -> ../random
│   │   ├── 189:0 -> ../bus/usb/001/001
│   │   ├── 189:128 -> ../bus/usb/002/001
│   │   ├── 1:9 -> ../urandom
│   │   ├── 202:0 -> ../cpu/0/msr
│   │   ├── 203:0 -> ../cpu/0/cpuid
│   │   ├── 21:0 -> ../sg0
│   │   ├── 21:1 -> ../sg1
│   │   ├── 226:0 -> ../dri/card0
│   │   ├── 226:64 -> ../dri/controlD64
│   │   ├── 250:0 -> ../usbmon0
│   │   ├── 250:1 -> ../usbmon1
│   │   ├── 250:2 -> ../usbmon2
│   │   ├── 251:0 -> ../bsg/1:0:0:0
│   │   ├── 251:1 -> ../bsg/2:0:0:0
│   │   ├── 253:0 -> ../rtc0
│   │   ├── 29:0 -> ../fb0
│   │   ├── 4:0 -> ../tty0
│   │   ├── 4:1 -> ../tty1
│   │   ├── 4:10 -> ../tty10
│   │   ├── 4:11 -> ../tty11
│   │   ├── 4:12 -> ../tty12
│   │   ├── 4:13 -> ../tty13
│   │   ├── 4:14 -> ../tty14
│   │   ├── 4:15 -> ../tty15
│   │   ├── 4:16 -> ../tty16
│   │   ├── 4:17 -> ../tty17
│   │   ├── 4:18 -> ../tty18
│   │   ├── 4:19 -> ../tty19
│   │   ├── 4:2 -> ../tty2
│   │   ├── 4:20 -> ../tty20
│   │   ├── 4:21 -> ../tty21
│   │   ├── 4:22 -> ../tty22
│   │   ├── 4:23 -> ../tty23
│   │   ├── 4:24 -> ../tty24
│   │   ├── 4:25 -> ../tty25
│   │   ├── 4:26 -> ../tty26
│   │   ├── 4:27 -> ../tty27
│   │   ├── 4:28 -> ../tty28
│   │   ├── 4:29 -> ../tty29
│   │   ├── 4:3 -> ../tty3
│   │   ├── 4:30 -> ../tty30
│   │   ├── 4:31 -> ../tty31
│   │   ├── 4:32 -> ../tty32
│   │   ├── 4:33 -> ../tty33
│   │   ├── 4:34 -> ../tty34
│   │   ├── 4:35 -> ../tty35
│   │   ├── 4:36 -> ../tty36
│   │   ├── 4:37 -> ../tty37
│   │   ├── 4:38 -> ../tty38
│   │   ├── 4:39 -> ../tty39
│   │   ├── 4:4 -> ../tty4
│   │   ├── 4:40 -> ../tty40
│   │   ├── 4:41 -> ../tty41
│   │   ├── 4:42 -> ../tty42
│   │   ├── 4:43 -> ../tty43
│   │   ├── 4:44 -> ../tty44
│   │   ├── 4:45 -> ../tty45
│   │   ├── 4:46 -> ../tty46
│   │   ├── 4:47 -> ../tty47
│   │   ├── 4:48 -> ../tty48
│   │   ├── 4:49 -> ../tty49
│   │   ├── 4:5 -> ../tty5
│   │   ├── 4:50 -> ../tty50
│   │   ├── 4:51 -> ../tty51
│   │   ├── 4:52 -> ../tty52
│   │   ├── 4:53 -> ../tty53
│   │   ├── 4:54 -> ../tty54
│   │   ├── 4:55 -> ../tty55
│   │   ├── 4:56 -> ../tty56
│   │   ├── 4:57 -> ../tty57
│   │   ├── 4:58 -> ../tty58
│   │   ├── 4:59 -> ../tty59
│   │   ├── 4:6 -> ../tty6
│   │   ├── 4:60 -> ../tty60
│   │   ├── 4:61 -> ../tty61
│   │   ├── 4:62 -> ../tty62
│   │   ├── 4:63 -> ../tty63
│   │   ├── 4:64 -> ../ttyS0
│   │   ├── 4:65 -> ../ttyS1
│   │   ├── 4:66 -> ../ttyS2
│   │   ├── 4:67 -> ../ttyS3
│   │   ├── 4:7 -> ../tty7
│   │   ├── 4:8 -> ../tty8
│   │   ├── 4:9 -> ../tty9
│   │   ├── 5:0 -> ../tty
│   │   ├── 5:1 -> ../console
│   │   ├── 5:2 -> ../ptmx
│   │   ├── 7:0 -> ../vcs
│   │   ├── 7:1 -> ../vcs1
│   │   ├── 7:128 -> ../vcsa
│   │   ├── 7:129 -> ../vcsa1
│   │   ├── 7:130 -> ../vcsa2
│   │   ├── 7:131 -> ../vcsa3
│   │   ├── 7:132 -> ../vcsa4
│   │   ├── 7:133 -> ../vcsa5
│   │   ├── 7:134 -> ../vcsa6
│   │   ├── 7:2 -> ../vcs2
│   │   ├── 7:3 -> ../vcs3
│   │   ├── 7:4 -> ../vcs4
│   │   ├── 7:5 -> ../vcs5
│   │   └── 7:6 -> ../vcs6
│   ├── console
│   ├── core -> /proc/kcore
│   ├── cpu
│   │   ├── 0
│   │   │   ├── cpuid
│   │   │   └── msr
│   │   └── microcode
│   ├── cpu_dma_latency
│   ├── cuse
│   ├── disk
│   │   ├── by-id
│   │   │   ├── ata-VBOX_CD-ROM_VB2-01700376 -> ../../sr0
│   │   │   ├── ata-VBOX_HARDDISK_VB72e5fffe-dea14268 -> ../../sda
│   │   │   ├── ata-VBOX_HARDDISK_VB72e5fffe-dea14268-part1 -> ../../sda1
│   │   │   ├── ata-VBOX_HARDDISK_VB72e5fffe-dea14268-part2 -> ../../sda2
│   │   │   └── ata-VBOX_HARDDISK_VB72e5fffe-dea14268-part3 -> ../../sda3
│   │   ├── by-label
│   │   │   └── ol_docker -> ../../sda3
│   │   ├── by-path
│   │   │   ├── pci-0000:00:01.1-ata-2.0 -> ../../sr0
│   │   │   ├── pci-0000:00:0d.0-ata-1.0 -> ../../sda
│   │   │   ├── pci-0000:00:0d.0-ata-1.0-part1 -> ../../sda1
│   │   │   ├── pci-0000:00:0d.0-ata-1.0-part2 -> ../../sda2
│   │   │   └── pci-0000:00:0d.0-ata-1.0-part3 -> ../../sda3
│   │   └── by-uuid
│   │       ├── 5309c176-f2b1-48fd-825c-e16d13c08bfe -> ../../sda1
│   │       ├── 808345fa-bfd5-410a-bf4f-a71247667d0f -> ../../sda3
│   │       └── e515d445-dffd-4d36-bf3f-7df68c2c35bd -> ../../sda2
│   ├── dri
│   │   ├── card0
│   │   └── controlD64
│   ├── fb0
│   ├── fd -> /proc/self/fd
│   ├── full
│   ├── fuse
│   ├── hpet
│   ├── hugepages
│   │   └── libvirt
│   │       └── qemu
│   ├── hwrng
│   ├── initctl -> /run/systemd/initctl/fifo
│   ├── input
│   │   ├── by-path
│   │   │   ├── pci-0000:00:04.0-event-mouse -> ../event4
│   │   │   ├── platform-i8042-serio-0-event-kbd -> ../event2
│   │   │   ├── platform-i8042-serio-1-event-mouse -> ../event3
│   │   │   ├── platform-i8042-serio-1-mouse -> ../mouse0
│   │   │   └── platform-pcspkr-event-spkr -> ../event6
│   │   ├── event0
│   │   ├── event1
│   │   ├── event2
│   │   ├── event3
│   │   ├── event4
│   │   ├── event5
│   │   ├── event6
│   │   ├── mice
│   │   └── mouse0
│   ├── kmsg
│   ├── log
│   ├── loop-control
│   ├── lp0
│   ├── lp1
│   ├── lp2
│   ├── lp3
│   ├── mapper
│   │   └── control
│   ├── mcelog
│   ├── mem
│   ├── memory_bandwidth
│   ├── mqueue
│   ├── net
│   │   └── tun
│   ├── network_latency
│   ├── network_throughput
│   ├── null
│   ├── nvram
│   ├── port
│   ├── ppp
│   ├── ptmx
│   ├── pts
│   │   └── ptmx
│   ├── random
│   ├── raw
│   │   └── rawctl
│   ├── rtc -> rtc0
│   ├── rtc0
│   ├── sda
│   ├── sda1
│   ├── sda2
│   ├── sda3
│   ├── sg0
│   ├── sg1
│   ├── shm
│   ├── snapshot
│   ├── snd
│   │   ├── by-path
│   │   │   └── pci-0000:00:05.0 -> ../controlC0
│   │   ├── controlC0
│   │   ├── pcmC0D0c
│   │   ├── pcmC0D0p
│   │   ├── pcmC0D1c
│   │   ├── seq
│   │   └── timer
│   ├── sr0
│   ├── stderr -> /proc/self/fd/2
│   ├── stdin -> /proc/self/fd/0
│   ├── stdout -> /proc/self/fd/1
│   ├── tty
│   ├── tty0
│   ├── tty1
│   ├── tty10
│   ├── tty11
│   ├── tty12
│   ├── tty13
│   ├── tty14
│   ├── tty15
│   ├── tty16
│   ├── tty17
│   ├── tty18
│   ├── tty19
│   ├── tty2
│   ├── tty20
│   ├── tty21
│   ├── tty22
│   ├── tty23
│   ├── tty24
│   ├── tty25
│   ├── tty26
│   ├── tty27
│   ├── tty28
│   ├── tty29
│   ├── tty3
│   ├── tty30
│   ├── tty31
│   ├── tty32
│   ├── tty33
│   ├── tty34
│   ├── tty35
│   ├── tty36
│   ├── tty37
│   ├── tty38
│   ├── tty39
│   ├── tty4
│   ├── tty40
│   ├── tty41
│   ├── tty42
│   ├── tty43
│   ├── tty44
│   ├── tty45
│   ├── tty46
│   ├── tty47
│   ├── tty48
│   ├── tty49
│   ├── tty5
│   ├── tty50
│   ├── tty51
│   ├── tty52
│   ├── tty53
│   ├── tty54
│   ├── tty55
│   ├── tty56
│   ├── tty57
│   ├── tty58
│   ├── tty59
│   ├── tty6
│   ├── tty60
│   ├── tty61
│   ├── tty62
│   ├── tty63
│   ├── tty7
│   ├── tty8
│   ├── tty9
│   ├── ttyS0
│   ├── ttyS1
│   ├── ttyS2
│   ├── ttyS3
│   ├── uhid
│   ├── uinput
│   ├── urandom
│   ├── usbmon0
│   ├── usbmon1
│   ├── usbmon2
│   ├── vboxguest
│   ├── vboxuser
│   ├── vcs
│   ├── vcs1
│   ├── vcs2
│   ├── vcs3
│   ├── vcs4
│   ├── vcs5
│   ├── vcs6
│   ├── vcsa
│   ├── vcsa1
│   ├── vcsa2
│   ├── vcsa3
│   ├── vcsa4
│   ├── vcsa5
│   ├── vcsa6
│   ├── vfio
│   │   └── vfio
│   ├── vga_arbiter
│   ├── vhci
│   ├── vhost-net
│   └── zero

```

এটি একটি লিনাক্স সিস্টেমের একটি অংশ মাত্র আপনাদের সামনে তুলে ধরা হয়েছে। সম্পূর্ণ ফাইলটি আপনি এখন থেকে ডাউনলোড করতে পারেন - [File Tree](https://raw.githubusercontent.com/Palash90/misc/master/*nix/file_tree)

আপনি যদি ভালো করে লক্ষ্য করেন, তাহলে বুঝতে পারবেন, ফাইল সিস্টেম একটি জায়গায় শুরু হয়েছে এবং সেটির অনেক শাখা প্রশাখা আছে। লিনাক্সের ক্ষেত্রে সমস্ত কিছুই একটি জায়গায় শুরু হয়, সাধারণত সেটিকে root বলা হয়। আমরা ফাইল সিস্টেমের এই বিশেষ বিন্যাসকে একটি উল্টোনো গাছের সাথে তুলনা করতে পারি, যেখানে একটি মূল কান্ড থেকে বিভিন্ন শাখা প্রশাখা বিন্যস্ত হয়েছে এবং একটি শাখার এক বা একাধিক প্রশাখা থাকতে পারে এবং সেই সমস্ত শাখা/প্রশাখার থাকতে পারে অন্যান্য প্রশাখা এবং পাতা। অনেকটা নিচের ছবিটির মতো -
![image](https://3.bp.blogspot.com/-j_mUuvYblB8/WaBSf-ZPi1I/AAAAAAAAHbQ/ytZ_CNzHV7wpMnthDUGz4C8CD7nNdV7YACLcBGAs/s1600/inverted%2Btree.jpg)

এই বার আমাদের উদাহরণের ক্ষেত্রে মূল কান্ডটি হলো / (root) এবং মূল কান্ড থেকে  আসা তিনটি শাখা হলো bin, boot, dev এবং পুনরায় boot শাখাটির একটি প্রশাখা হলো grub2 এবং dev শাখাটির প্রশাখা হলো autofs, bsg, bus, cdrom ইত্যাদি এবং পুনরায় এই সমস্ত প্রশাখা গুলির বিভিন্ন প্রশাখা ও পাতার বিন্যাস আছে। একটু সময় নিয়ে দেখলে বুঝতে পারবেন,লিনাক্স সিস্টেমের মধ্যে অসংখ্য ফাইল বিভিন্ন ভাবে বিন্যস্ত হলেও সমস্ত কিছু শেষ পর্যন্ত / অর্থাৎ root এ গিয়ে শেষ হয় এবং আমরা যেকোনো সময়ে root এর যেকোনো একটি প্রশাখার মধ্যে অবস্থান করছি।
এই সমস্ত প্রশাখা গুলিকে লিনাক্সের ভাষায় বলা হয় ডিরেক্টরি (directory) এবং পাতা গুলি হলো ফাইল (file)। এইবার আমরা আগের মতোই একটি চিত্রের সাহায্যে ব্যাপারটিকে দেখে নেবার চেষ্টা করবো -
![image](https://2.bp.blogspot.com/-IfWyg7_cB04/WaBVgabvjsI/AAAAAAAAHbc/eT6teb38TpQ0dOfgGa30LVkP2lniCyfHACLcBGAs/s1600/FS-layout.png)

যেকোনো নতুন ইউজারের পক্ষে ফাইল সিস্টেমের চিত্রটি মানসিক ভাবে উপস্থাপন করা একটু অসুবিধার কিন্তু কয়েকদিন অভ্যাস করলে ব্যাপারটি অনেক সরল হয়ে যাবে।

আসুন এবার আমরা ফাইল সিস্টেম সম্পর্কিত কয়েকটি সাধারণ কম্যান্ড জেনে নি, যাতে করে আমরা ফাইল সিস্টেমে যেকোনো জায়গায় যাওয়া আসা (navigation/ন্যাভিগেশন) করতে পারি।

সব থেকে প্রথমে যেটা মাথায় আসে সেটা হচ্ছে, আমরা বর্তমানে কোন ডিরেক্টরিতে অবস্থান করছি, সেটি জানার জন্যে আপনাকে ব্যবহার করতে হবে **pwd কম্যান্ড, আসুন দেখে নি এটি কি কাজ করে -

```
totan@Home-Computer ~ $ pwd
/home/totan

```

এটি বোঝাচ্ছে যে আমি totan ইউজারের হোম ডিরেক্টরিতে অবস্থান করছি। সাধারণত হোম ডিরেক্টরি বলতে সেই বিশেষ ডিরেক্টরীটিকে বোঝায় যে ডিরেক্টরিতে একটি ইউজার লগইন করার পর প্রথম অবস্থান করে। আমি এখানে আরো কয়েকটি ডিরেক্টরিতে গিয়ে **pwd কম্যান্ড রান করবো -

```
totan@Home-Computer / $ pwd
/
totan@Home-Computer /mnt/sda3/Media/Audio $ pwd
/mnt/sda3/Media/Audio
totan@Home-Computer /home/palash $ pwd
/home/palash
totan@Home-Computer /dev $ pwd
/dev
totan@Home-Computer /boot/grub $ pwd
/boot/grub

```

আপনি প্রত্যেকটি আউটপুট দেখলে বুঝতে পারবেন, আমি বিভিন্ন ডিরেক্টরি থেকে **pwd কমান্ডটি রান করে ভিন্ন ভিন্ন ডিরেক্টরি দেখতে পেয়েছি কিন্তু সমস্ত ক্ষেত্রেই যেকোনো ডিরেক্টরি শুরু হয়েছে **root (/) থেকে এবং বিভিন্ন শাখা প্রশাখা ভ্রমণ করে আমরা যেকোনো একটি ডিরেক্টরির মধ্যে অবস্থান করছি এবং আমরা যেকোনো সময়ে একটি না একটি ডিরেক্টরির মধ্যে অবশ্যই অবস্থান করছি। এবং আপনি নিশ্চয় এটা ধারণা করতে পেরেছেন যে **/ চিহ্নটি বোঝায় যে **/ এর পরের ডিরেক্টরিটি **/ এর আগের ডিরেক্টরির মধ্যে অবস্থান করছে। যেমন ধরুন **grub ডিরেক্টরিটি **boot ডিরেক্টরির মধ্যে অবস্থান করছে এবং **boot ডিরেক্টরিটি অবস্থান করছে **/ এর মধ্যে বা অন্য্ ভাবে বললে বলা হয় **/ এর প্রশাখা **boot, যার প্রশাখা **grub। কম্পিউটারের ভাষায় আমরা বলবো **/ এর চাইল্ড ডিরেক্টরি **boot, যার চাইল্ড ডিরেক্টরি **grub এবং **grub এর প্যারেন্ট ডিরেক্টরি **boot যার প্যারেন্ট ডিরেক্টরি **/ বা **root।

এতক্ষন আমরা দেখলাম ফাইল সিস্টেমের বিভিন্ন শাখা প্রশাখা কিভাবে বিন্যস্ত থাকে এবং কিভাবে আমরা দেখতে পারি আমরা কোন ডিরেক্টরিতে অবস্থান করছি। এর পরে আমরা দেখে নেবো কিভাবে আমরা বিভিন্ন ডিরেক্টরিতে নেভিগেট করতে পারি এবং শিখে নেব ডিরেক্টরি সম্পর্কিত কিছু কমান্ড, সাথে সাথে জেনে নেব কিভাবে আমরা দেখতে পারি একটি ডিরেক্টরিতে কি কি চাইল্ড ডিরেক্টরি এবং ফাইল বর্তমান এবং শিখে নেব কিভাবে কিছু ফাইল তৈরি করা যায়।
