---
title: "Introduction to File System"
date: "2017-07-16T12:22:00+05:30"
recovered_from: "web.archive.org (snapshot 20191021045024)"
---

In earlier section, we have run some very simple commands, knew how they work; we also performed some common mathematical calculations ​​using shell. Now, we will see how we can work with files on Unix-Like systems. 
**Those who used Windows or Mac OS before using Unix/Linux or have used an Android phone or iPhone/iPad, file is not new to those, but those who have not used any such device before, they can have this question, "**what is this file and what does it do? Is it something like the paper files used in office or what?". We will now try to figure out what file is -
The file is a combination of **bytes. So now your question boils down to, **"what is this byte now?". To know that we have to dig deeper. In fact, information is exchanged with a computer (or within a set of computers) in binary format and data is also stored in binary. Binary is nothing but a number representation system where any information boils down to **0 and **1. This unit is called a bit, and combination of 8 bits is called a byte. Let's try to understand the matter with an example - 
Suppose a number **'*128' is to be given to computer as input, you type 128 on the keyboard but the computer will take it as ***10000000. In the same way, if you have to give input '**A', you will type ***A on the keyboard but input is given as ***1000001 to the computer. 
**If you are curious about binary number system, you can read this - 
[binary number](https://en.wikipedia.org/wiki/Binary_number)
**Generally any file stores information in this form. So, in a high level, we can say that the file stores information in memory in binary format and as and when needed the stored information is fetched from memory for calculation. In short, the file is a container that helps in the calculation*** of computer by saving information in binary format.
**In fact, in any Unix/Linux system file is very important, because everything on the Unix/Linux system is either a file or process (we will know about the process later) and when you run the system, it reads various type of configuration files. Here we will check how we can work with different files.
**All files in Unix/Linux are arranged in different directories, let's see how. We'll put a small picture here -

```
/
├── bin -> usr/bin
├── boot
├ ├── config-3.10.0-327.22.2.el7.x86_64
│   ├── config-4.1.12-37.5.1.el7uek.x86_64
│ ├── Grub2
│   │   ├── device.map
│   │   ├── fonts
│   │   │   └── unicode.pf2
│   │   ├── grub.cfg
│ │ ├── grubenv
│   │   ├── i386-pc
│   │   │   ├── acpi.mod
│   │   │   ├── adler32.mod
│ │ │ ├── affs.mod
│   │   │   ├── afs.mod
│   │   │   ├── ahci.mod
│   │   │   ├── all_video.mod
│ │ │ ├── aout.mod
│   │   │   ├── archelp.mod
│ │ │ ├──
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
│ │ │ ├── cat
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
│ │ │ ├── efiemu32.o
│ │ │ ├── efiemu64.o
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
│ │ │ ├── gcry_crc.mod
│ │ │ ├── gcry_des.mod
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
│ │ │ ├── gcry_tiger.mod
│   │   │   ├── gcry_twofish.mod
│   │   │   ├── gcry_whirlpool.mod
│   │   │   ├── gdb.mod
│ │ │ ├── geli.mod
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
│ │ │ ├── luks.mod
│   │   │   ├── lvm.mod
│   │   │   ├── lzopio.mod
│   │   │   ├── macbless.mod
│   │   │   ├── macho.mod
│ │ │ ├── mda_text.mod
│   │   │   ├── mdraid09_be.mod
│   │   │   ├── mdraid09.mod
│   │   │   ├── mdraid1x.mod
│ │ │ ├── memdisk.mod
│   │   │   ├── memrw.mod
│ │ │ ├── minicmd.mod
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
│ │ │ ├── mpi.mod
│   │   │   ├── msdospart.mod
│ │ │ ├── multiboot2.mod
│   │   │   ├── multiboot.mod
│   │   │   ├── nativedisk.mod
│   │   │   ├── net.mod
│   │   │   ├── newc.mod
│   │   │   ├── nilfs2.mod
│   │   │   ├── normal.mod
│   │   │   ├── ntfscomp.mod
│   │   │   ├── ntfs.mod
│   │   │   ├── ntldr.mod
│ │ │ ├ - odc.mod
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
│ │ │ ├── pata.mod
│   │   │   ├── pbkdf2.mod
│ │ │ ├── pbkdf2_test.mod
│   │   │   ├── pcidump.mod
│ │ │ ├── pci.mod
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
│ │ │ ├── sfs.mod
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
│ │ │ ├── testload.mod
│ │ │ ├── test.mod
│ │ │ ├── testspeed.mod
│   │   │   ├── tftp.mod
│   │   │   ├── tga.mod
│   │   │   ├── time.mod
│   │   │   ├── trig.mod
│   │   │   ├── tr.mod
│   │   │   ├── truecrypt.mod
│   │   │   ├── true.mod
│ │ │ ├── execution mode
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
│ │ │ ├── videoinfo.mod
│   │   │   ├── video.lst
│ │ │ ├───
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
│ │ │ ├─ ast.mo
│ │ │ ├─│ ca.mo
│ │ │ ├─ da.mo
│ │ │ ├── de_CH.mo
De@hebrew.mo
│ │ │ ├─ de.mo
En_arabic.mo
│ │ │ ├── en@cyrillic.mo
En@greek.mo
│ │ │ ├── en@hebrew.mo
│ │ │ ├─ en.mo
│ │ │ ├── en@piglatin.mo
En *
│ │ │ ├─ eo.mo
He is
│ │ ├── │ - fi .mo
│ │ │ ├─ fr mo
│ │ │ ├─ gl.mo
├ ├── hu.mo
├ ├── id.mo
│   │   │   ├── it.mo
│ │ │ ├─ ja.mo
│ │ │ ├─────────────
Nl.mo
│ │ │ ├ ─ ─ pa.mo
│ │ │ ├─ pl.
│ │ │ ├── pt_BR.mo
│ │ │ ├─ ru.mo
│ │ │ ├── sl.mo
│ │ │ ├── sv.mo
│ │ │ ├─ tr.mo
│   │   │   ├── uk.mo
│ │ │ ├─ vi.mo
│   │   │   ├── zh_CN.mo
│   │   │   └── zh_TW.mo
│   │   └── themes
│   │       └── system
│   ├── initramfs-0-rescue-8d9a1061e6b348e68bab0fe7454ab730.img
├├─ in
│   ├── initramfs-4.1.12-37.5.1.el7uek.x86_64.img
│   ├── initramfs-4.1.12-37.5.1.el7uek.x86_64kdump.img
│   ├── initrd-plymouth.img
├ ├── symvers-3.10.0-327.22.2.el7.x86_64.gz
│   ├── symvers-4.1.12-37.5.1.el7uek.x86_64.gz
├ ├── System.map-3.10.0-327.22.2.el7.x86_64
│   ├── System.map-4.1.12-37.5.1.el7uek.x86_64
│   ├── vmlinuz-0-rescue-8d9a1061e6b348e68bab0fe7454ab730
├ ├── vmlinuz-3.10.0-327.22.2.el7.x86_64
│   └── vmlinuz-4.1.12-37.5.1.el7uek.x86_64
├── dev
│   ├── autofs
│   ├── block
│   │   ├── 11:0 -> ../sr0
│ │ ├── 8: 0 -> ../sda
│   │   ├── 8:1 -> ../sda1
│   │   ├── 8:2 -> ../sda2
└ └── 8: 3 -> ../sda3
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
│ │ ├── 1: 1 -> ../mem
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
├ ├── 202: 0 -> ../cpu/0/msr
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
│ │ ├── 7: 0 -> ../vcs
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
│ │ │ ├── cpuid
└ └── msr
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
└ └── ol_docker -> ../../sda3
│   │   ├── by-path
│ │ │ ├── pci-0000: 00: 01.1-ata-2.0 -> ../../sr0
│   │   │   ├── pci-0000:00:0d.0-ata-1.0 -> ../../sda
│ │ │ ├── pci-0000: 00: 0d.0-ata-1.0-part1 -> ../../sda1
│ │ │ ├── pci-0000: 00: 0d.0-ata-1.0-part2 -> ../../sda2
└ └── pci-0000: 00: 0d.0-ata-1.0-part3 -> ../../sda3
│   │   └── by-uuid
│   │       ├── 5309c176-f2b1-48fd-825c-e16d13c08bfe -> ../../sda1
├ ├── 808345fa-bfd5-410a-bf4f-a71247667d0f -> ../../sda3
│   │       └── e515d445-dffd-4d36-bf3f-7df68c2c35bd -> ../../sda2
│   ├── dri
│ │ ├── card0
│   │   └── controlD64
│ ├── fb0
│   ├── fd -> /proc/self/fd
│   ├── full
│   ├── fuse
│ ├── hpet
│ ├── hugepages
│   │   └── libvirt
│   │       └── qemu
│ ├── hurdle
│   ├── initctl -> /run/systemd/initctl/fifo
│   ├── input
│   │   ├── by-path
│   │   │   ├── pci-0000:00:04.0-event-mouse -> ../event4
│   │   │   ├── platform-i8042-serio-0-event-kbd -> ../event2
│   │   │   ├── platform-i8042-serio-1-event-mouse -> ../event3
│   │   │   ├── platform-i8042-serio-1-mouse -> ../mouse0
│   │   │   └── platform-pcspkr-event-spkr -> ../event6
│ │ ├── event0
│ │ ├── event1
│ │ ├── event2
│ │ ├── event3
│ │ ├── event4
│ │ ├── event5
│ │ ├── event6
│   │   ├── mice
│ │ └──────────────────
│ ├── kmsg
│   ├── log
│   ├── loop-control
│ ├── lp0
│ ├── lp1
│ ├── lp2
├ ├── lp3
│   ├── mapper
│   │   └── control
│ ├── mcelog
│ ├── mem
│   ├── memory_bandwidth
│ ├── mqueue
│   ├── net
│ │ └── do
│   ├── network_latency
│   ├── network_throughput
│   ├── null
│   ├── nvram
│   ├── port
│   ├── ppp
│ ├── ptmx
│   ├── pts
│ │ └── ptmx
│   ├── random
│   ├── raw
│ │ └── rawctl
│   ├── rtc -> rtc0
│ ├── rtc0
│ ├── sda
│ ├── sda1
├ ├── sda2
├ ├── sda3
│ ├── sg0
│ ├── sg1
│ ├── shm
│   ├── snapshot
│   ├── snd
│   │   ├── by-path
│   │   │   └── pci-0000:00:05.0 -> ../controlC0
│   │   ├── controlC0
│ │ ├── pcmC0D0c
│ │ ├── pcmC0D0p
│ │ ├── pcmC0D1c
│   │   ├── seq
│   │   └── timer
│ ├── sr0
│   ├── stderr -> /proc/self/fd/2
│   ├── stdin -> /proc/self/fd/0
│   ├── stdout -> /proc/self/fd/1
│   ├── tty
│ ├── tty0
│ ├── tty1
│ ├── tty10
│ ├── tty11
│ ├── tty12
│ ├── tty13
│ ├── tty14
│ ├── tty15
│ ├── tty16
│ ├── tty17
│ ├── tty18
│ ├── tty19
│ ├── tty2
│ ├── tty20
│ ├── tty21
│ ├── tty22
│ ├── tty23
│ ├── tty24
│ ├── tty25
│ ├── tty26
│ ├── tty27
│ ├── tty28
│ ├── tty29
│ ├── tty3
│ ├── tty30
│ ├── tty31
│ ├── tty32
│ ├── tty33
│ ├── tty34
│ ├── tty35
│ ├── tty36
│ ├── tty37
│ ├── tty38
│ ├── tty39
│ ├── tty4
│ ├── tty40
│ ├── tty41
│ ├── tty42
│ ├── tty43
│ ├── tty44
│ ├── tty45
│ ├── tty46
│ ├── tty47
│ ├── tty48
│ ├── tty49
│ ├── tty5
│ ├── tty50
│ ├── tty51
│ ├── tty52
│ ├── tty53
│ ├── tty54
│ ├── tty55
│ ├── tty56
│ ├── tty57
│ ├── tty58
│ ├── tty59
│ ├── tty6
│ ├── tty60
│ ├── tty61
│ ├── tty62
│ ├── tty63
│ ├── tty7
│ ├── tty8
│ ├── tty9
│ ├── ttyS0
│ ├── ttyS1
│ ├── ttyS2
│ ├── ttyS3
│ ├── uhid
│ ├── uinput
│   ├── urandom
│ ├── usbmon0
│ ├── usbmon1
│ ├── usbmon2
│ ├── vboxguest
│ ├── vboxuser
│   ├── vcs
│ ├── vcs1
│ ├── vcs2
│ ├── vcs3
│ ├── vcs4
│ ├── vcs5
│ ├── vcs6
You
│ ├── vcsa1
│ ├── vcsa2
│ ├── vcsa3
│ ├── vcsa4
│ ├── vcsa5
│ ├── vcsa6
│ ├── vfio
│ │ └── vfio
│   ├── vga_arbiter
│ ├── vhci
│   ├── vhost-net
│   └── zero

```

**This is just a part of a Unix/Linux system that is presented to you. You can download the complete file here - [File Tree](https://raw.githubusercontent.com/Palash90/misc/master/*nix/file_tree)**If you notice closely, you can understand that the file system has started in one place and it has many branches in it. In Unix/Linux, everything starts at one place, usually called **root. We can compare this pattern of file system to an inverted tree, where different branches are forked from an original stem and one branch may have one or more branches and those branches may contain other branches and leaves. Much like the image below -
![image](https://3.bp.blogspot.com/-j_mUuvYblB8/WaBSf-ZPi1I/AAAAAAAAHbQ/ytZ_CNzHV7wpMnthDUGz4C8CD7nNdV7YACLcBGAs/s1600/inverted%2Btree.jpg)
**This time, we compare this image with our previous file system hierarchy, there are three branches (**bin, boot, dev) from **root (**/), analogous to the main stem and in turn **boot has a branch **grub2 and **dev has the branches **autofs, bsg, bus, cdrom etc. There are different types of branch and leaf arrangements in Unix/Linux File System. Looking into it carefully, you can understand every branch or sub-branch started from root and at any point of time, you must be in one of the branches or sub-branches of root.

These branches are known as directory in Unix/Linux and the leaves are known as Files. Now, we will try to understand this with the help of another picture -
![image](https://2.bp.blogspot.com/-IfWyg7_cB04/WaBVgabvjsI/AAAAAAAAHbc/eT6teb38TpQ0dOfgGa30LVkP2lniCyfHACLcBGAs/s1600/FS-layout.png)
**For a new user, it is hectic to mentally represent the file system, but if you practice for a few days it will be very straightforward. **Let us now know some common commands related to file system, so that we can navigate anywhere within the file system. **The first thing that comes to mind is to know which directory we are currently in. To know this, you have to use the **pwd command, let's see what it works -

```
totan@Home-Computer ~ $ pwd
/home/totan

```

This implies that I'm in the home directory of user **totan. Generally, the home directory refers to that particular directory that is the first place a user steps after logging in. I will run the **pwd command in a few more directories -

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

If you see each output, you can see in every case, any directory starts from **root (/) and we are staying in any of the different branches of it and at any time we are within a directory. And you've surely understood that the **/ symbol indicates that the directory in the right side of the **/ is located in the directory in the left side of **/. For example, the **grub directory is located in the **boot directory and the **boot directory is located in **/ (aka** root) or the other way to represent it is **boot is a child directory of **root (/) and **grub is child of **boot. Or the other way around, **root is the parent directory of **boot while **boot is the parent directory of **grub.

Confusing, right ?
Re-read the file structure in this article and the downloaded file and you will understand.
**Here we saw how the file system looks like in the OS and how you can know where you are currently located. Next we'll take a step further and will create, delete, rename files and directories, move around here and there, will see how we can know what is inside a directory etc. in our subsequent discussions.**
Foot Notes -
***If  you are surprised by the term calculation, keep in mind that whatever a computer does is some basic mathematical calculation from a predefined set; over and over again in a mind boggling speed. Don't get fooled by he term artificial intelligence. Even a computer decides lot of stuff on its own, it still does this by the combination of a small subset of basic arithmetic calculations. It is really surprising to know that using a subset of arithmetic calculations a piece of silicon wafer becomes intelligent on its own to the extent of deciding itself what to do or what not to. If curiosity grows, read this article - [Computer performs basic mathematical calculations](https://engineering.mit.edu/engage/ask-an-engineer/how-do-computers-perform-complex-mathematical-operations/).

<div class='
