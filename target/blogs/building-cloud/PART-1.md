# My ₹12,000 Mistake: A Lesson in Thermal Failure and Data Backups

In my recent journey of hosting my own private cloud, I made a massive mistake. It was a mistake that taught me several lessons the hard way. It cost me ₹12000/-. I am writing this, so you don't make the same error.

## The Story
In an attempt to download all my data from cloud, I needed a large storage. I purchased a WD My Passport 2 TB almost three months ago for this purpose.

Once the disk was attached to my machine, I got distracted by an AI Project and the HDD essentially became a storage for the open source models.

That streak went on for almost two months. Then the urgency hit as my Google One subscription is ending in a month.

I quickly jot down the requirements and purchased new hardware.

## The Plan
I planned for a 3-2-1 backup strategy. 

- 3 copies of data
- 2 different types of media 
- 1 offsite copy

Plan was simple: copy all data on all three devices and move one copy to offsite.

## The Mistake: A Heat Trap
Then came the costly mistake, I tried to copy data over wifi and it worked at a very slow speed. Then it clicked me that I can put the USB HDD in the PC and then start `rsync`. After a few GB of data copy, the failure started. I ignored and restarted the rsync quite a few times. It kept on failing.

Almost after 18 hours, when nothing clicked. I dug into the issue. When I checked the PC, I found it. I placed the HDD on top of a Pi Enclosure and the device was very hot touch. I immediately pulled it apart, placed in front of a high speed fan, re-attached again. But the damage was done. The fault was hardwired into the circuit board itself.

I followed a few steps with `ddrescue`, `smartctl` but `lsblk` did not show my drive no matter what.

And there it was my ₹12000/- HDD, completely silent. A deep agony covered me, a deep sigh and it was the time to accept.

## Conclusion

I learnt the hard way about the Thermal Failures of Hard Disks. A real practical lesson, rather than a theoretical one.

Lesson learnt, always check for clearance, air gap and proper ventilation before you start copying data on a drive.

