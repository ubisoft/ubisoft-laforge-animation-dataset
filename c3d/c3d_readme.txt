ABOUT LaFAN1_C3D.zip:

	Note that C3D files were exported from source files BEFORE any retargeting, as opposed to the BVH files, which were exported AFTER retargeting to a common skeleton.

	This means that even though the motion and timing match between C3D and BVH, the correspondence is not perfect, e.g. with scale differences.



	As opposed to the BVH files, C3D files contain multiple subjects. This is why the subject ID is not in the filename anymore.

	The C3D filenames indicate the generic motion category, and matches motion categories found in the BVH filenames.

	The subject IDs can be found in the contents of the C3D (e.g. in the scene explorer of Motion Builder).



	There are two excpetions / mismatch between the BVH and C3D filename correspondence:
	 
	 - The file pushAndFall2.c3d contains the motion from fallAndGetUp3_subject1.bvh and push1_subject2.bvh that were split into different catagories when naming the BVHs.
	 
	 - The file fallAnGetUp2.c3d contains the additional motion from walk4_subject1.bvh.
	 
	 Despite the naming mismatches, all BVH motions are contained (and only contained once) in the C3Ds.
 