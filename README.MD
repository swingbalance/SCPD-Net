SCPD-Net: Siamese Correlation Pyramid Deformation Network for Chest X-Ray Image Registration
=
A fast chest X-rays images registration model based on deep learning

Intro
=
Thoraclogist spends lots of time to compare chest X-ray images(means one patient captures at different time) to find out lesions.

This model try to warp moving image to fixed image as similar as possible,
therefore doctor can easily find out the difference by comparing warped image and fixed image.

Evaluation
=
<table>
   <tr align="center" valign="center">
      <td></td>
      <th colspan=2>Deep learning</th>
      <th colspan=2>Traditional method</th>
   </tr>
   <tr align="center" valign="center">
      <td>Metrics</td>
      <th>VoxelMorph</th>
      <th>SCPD-Net(Ours)</th>
      <th>DisplacementField</th>
      <th>BSpline</th>
   </tr>
   <tr align="center" valign="center">
      <td>Dice</td>
      <td>0.9649</td>
      <th>0.9878</th>
      <td>0.9858</td>
      <td>0.9722</td>
   </tr>
   <tr align="center" valign="center">
      <td>IoU</td>
      <td>0.9324</td>
      <th>0.976</th>
      <td>0.9723</td>
      <td>0.9473</td>
   </tr>
   <tr align="center" valign="center">
      <td>MAD</td>
      <td>0.0438</td>
      <th>0.0436</th>
      <td>0.1123</td>
      <td>0.107</td>
   </tr>
   <tr align="center" valign="center">
      <td>MSE</td>
      <th>0.0042</th>
      <th>0.0042</th>
      <td>0.0256</td>
      <td>0.0175</td>
   </tr>
   <tr align="center" valign="center">
      <td>NCC</td>
      <td>0.9373</td>
      <th>0.9379</th>
      <td>0.807</td>
      <td>0.8949</td>
   </tr>
   <tr align="center" valign="center">
      <td>SSIM</td>
      <td>0.7742</td>
      <th>0.778</th>
      <td>0.6636</td>
      <td>0.6468</td>
   </tr>
   <tr align="center" valign="center">
      <td>PSNR(dB)</td>
      <td>24.1474</td>
      <th>24.1795</th>
      <td>16.407</td>
      <td>18.0868</td>
   </tr>
   <tr align="center" valign="center">
      <td>Time(secs/image)</td>
      <td>0.263</td>
      <th>0.2533</th>
      <td>16.5877</td>
      <td>12.2908</td>
   </tr>
</table>

Results
=
### Input pairs and warped results

<table>
   <tr align="center" valign="center">
      <td></td>
      <td colspan=2>Input</td>
      <td colspan=2>Output</td>
   </tr>
   <tr align="center" valign="center">
     <td> Patient's ID </td>
     <td> Moving image </td>
     <td> Fixed image </td>
     <th> SCPD-Net(Ours) </th>
     <td> DisplacementField</td>
   </tr>
   <tr align="center" valign="center">
     <td> 3 </td>
     <td><img src="images/Input_and_Warped/a1.png"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Input_and_Warped/a2.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Input_and_Warped/a3.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Input_and_Warped/a4.jpg"  alt="1" width = 153px height = 153px ></td>
   </tr> 
   <tr align="center" valign="center">
     <td> 25 </td>
     <td><img src="images/Input_and_Warped/b1.png"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Input_and_Warped/b2.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Input_and_Warped/b3.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Input_and_Warped/b4.jpg"  alt="1" width = 153px height = 153px ></td>
   </tr>
   <tr align="center" valign="center">
     <td> 27 </td>
     <td><img src="images/Input_and_Warped/c1.png"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Input_and_Warped/c2.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Input_and_Warped/c3.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Input_and_Warped/c4.jpg"  alt="1" width = 153px height = 153px ></td>
   </tr>
</table>

### Lung Contours

<table>
   <tr align="center" valign="center">
     <td> Patient's ID </td>
     <td> SCPD-Net(Ours) </td>
     <td> DisplacementField </td>
   </tr>
   <tr align="center" valign="center">
     <td> 3 </td>
     <td><img src="images/Contours/l1.png"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Contours/r1.png"  alt="1" width = 153px height = 153px ></td>
   </tr> 
   <tr align="center" valign="center">
     <td> 25 </td>
     <td><img src="images/Contours/l2.png"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Contours/r2.png"  alt="1" width = 153px height = 153px ></td>
   </tr>
   <tr align="center" valign="center">
     <td> 27 </td>
     <td><img src="images/Contours/l3.png"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Contours/r3.png"  alt="1" width = 153px height = 153px ></td>
   </tr>
</table>

Blue : moving image  
Red : warped image  
Green : fixed image  

### Comparison image and Overlay image

<table>
   <tr align="center" valign="center">
      <td></td>
      <td colspan=2>Comparison image</td>
      <td colspan=2>Overlay image</td>
   </tr>
   <tr align="center" valign="center">
     <td> Patient's ID </td>
     <td> SCPD-Net(Ours) </td>
     <td> DisplacementField </td>
     <th> SCPD-Net(Ours) </th>
     <td> DisplacementField</td>
   </tr>
   <tr align="center" valign="center">
     <td> 3 </td>
     <td><img src="images/Comparison_and_Overlay/a1.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Comparison_and_Overlay/a2.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Comparison_and_Overlay/a3.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Comparison_and_Overlay/a4.jpg"  alt="1" width = 153px height = 153px ></td>
   </tr> 
   <tr align="center" valign="center">
     <td> 25 </td>
     <td><img src="images/Comparison_and_Overlay/b1.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Comparison_and_Overlay/b2.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Comparison_and_Overlay/b3.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Comparison_and_Overlay/b4.jpg"  alt="1" width = 153px height = 153px ></td>
   </tr>
   <tr align="center" valign="center">
     <td> 27 </td>
     <td><img src="images/Comparison_and_Overlay/c1.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Comparison_and_Overlay/c2.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Comparison_and_Overlay/c3.jpg"  alt="1" width = 153px height = 153px ></td>
     <td><img src="images/Comparison_and_Overlay/c4.jpg"  alt="1" width = 153px height = 153px ></td>
   </tr>
</table>

Purple : warped image  
Green : fixed image

