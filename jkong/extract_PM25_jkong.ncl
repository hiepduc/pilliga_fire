load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/contributed.ncl"
load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW.ncl"
;load "$NCARG_ROOT/lib/ncarg/nclscripts/wrf/WRFUserARW_narclim.ncl"
;load "./gsn_shapefile_polylines.ncl"

  site=new((/22,2/),float)
  siteid=new((/22/), integer)

  site(0,0) = -33.89 ; Chullora
  site(0,1)= 151.05
  site(1,0) = -33.92 ; Earlwood
  site(1,1)= 151.13
  site(2,0) = -33.78 ; Lindfield
  site(2,1)= 151.15
  site(3,0) = -33.93 ; Randwick
  site(3,1)= 151.24
  site(4,0) = -33.86 ; Rozelle
  site(4,1)= 151.16
  site(5,0) = -33.79 ; Prospect
  site(5,1)= 150.91
  site(6,0) = -33.62 ; Richmond
  site(6,1)= 150.75
  site(7,0) = -33.79 ; St Marys
  site(7,1)= 150.77
  site(8,0) = -33.66 ; Vineyard
  site(8,1)= 150.85
  site(9,0) = -34.31 ; Bargo
  site(9,1)= 150.58
  site(10,0) = -33.92 ; Bringelly
  site(10,1)= 150.76
  site(11,0) = -34.04 ; Camden
  site(11,1)= 150.69
  site(12,0) = -34.07 ; Campbelltown West
  site(12,1)= 150.79
  site(13,0) = -33.93 ; Liverpool
  site(13,1)= 150.91
  site(14,0) = -34.05 ; Oakdale
  site(14,1)= 150.50
  site(15,0) = -34.58 ; Albion Park South
  site(15,1)= 150.78
  site(16,0) = -34.47 ; Kembla Grange
  site(16,1)= 150.82
  site(17,0) = -34.42 ; Wollongong
  site(17,1)= 150.89
  site(18,0) = -33.28 ; Wyong
  site(18,1)= 151.43
  site(19,0) = -32.80 ; Beresfield
  site(19,1)= 151.66
  site(20,0) = -32.93 ; Newcastle
  site(20,1)= 151.76
  site(21,0) = -32.89 ; Wallsend
  site(21,1)= 151.67


  siteid(0) = 190
  siteid(1) = 206
  siteid(2) = 70
  siteid(3) = 33
  siteid(4) = 39
  siteid(5) = 1148
  siteid(6) = 573
  siteid(7) = 760
  siteid(8) = 765
  siteid(9) = 574
  siteid(10) = 171
  siteid(11) = 2570
  siteid(12) = 2560
  siteid(13) = 107
  siteid(14) = 1570
  siteid(15) = 2921
  siteid(16) = 526
  siteid(17) = 500
  siteid(18) = 259   ; Wyong
  siteid(19) = 322
  siteid(20) = 300
  siteid(21) = 287

                      
;   jlat = (/-33.89, -33.92/) ; indices of desired grid points
;   ilon = (/151.05, 151.13/)
;   npts = dimsizes(jlat)
 
   npts = dimsizes(siteid)

;  diri = "/mnt/scratch3/nhannah/WRF-CMAQ/run/20fbd7af--2021-05-16T06/WRF/run/"
;  fname= "wrfout_d01_2019-12-23_00:00:00_dec23-31.nc"
;  diro = "./outpred/"
   diro = "/mnt/scratch_lustre/duch/runpillaga/jkong/"
;   f = addfile(diri+fname+".nc","r")
;   f = addfile(diri+fname,"r")
; dir = "/home/duch/mywrf/WRF/runfire201920/"
  dir = "/mnt/scratch_lustre/duch/runpillaga/jkong/"
; fili = systemfunc("cd "+dir+" ; ls  wrfout_d01_2023-12-18_00:00:00*.nc")
; fili = systemfunc("cd "+dir+" ; ls  wrfout_d02_2023-12-8to19.nc")
  fili = systemfunc("cd "+dir+" ; ls  wrfout_d02_2023-12-*:00:00")
  print(fili) ; all file names
  nfili = dimsizes(fili)


;sdata = new ((/nfili, npts/), string)

do nf=0,nfili-1
  f = addfile(dir+fili(nf), "r")

  print("File number  :  "  + nf + "  file name  " + fili(nf) )

; We generate plots, but what kind do we prefer?
; type = "X11"
; wks = gsn_open_wks(type,"plt_CTM_mumba")

; What times and how many time steps are in the data set?
; times  = a->wrt_step  ; get times in the file
;  ntimes = 72         ; number of times in the file



   Times= tostring(f->Times)
   ntim = dimsizes(Times)
   print(Times)

   pm25= f->PM2_5_DRY

; one grid point per ascii file

   sdata = new ((/ntim, nfili, npts/), string)

   res = True      
   res@returnInt = True

   do np=0,npts-1
;     point = wrf_user_ll_to_ij(f,ilon(np),jlat(np),res)   ; wrf_user_ll_to_ij(nc_file,lon,lat,opt)
      point = wrf_user_ll_to_ij(f,site(np,1),site(np,0),res)   ; wrf_user_ll_to_ij(nc_file,lon,lat,opt)
 
      x = point(1)
      y = point(0)
      sdata(:,nf,np) = Times+" "+sprintf("%9.2f",pm25(:,0,x,y))  ; ground level pm2.5
   end do 

end do

do np=0,npts-1
      ftxt = "PM25"+siteid(np)+".txt"
;     system("/bin/rm -f "+diro+ftxt)
;     shead = "Time " + "PM2.5 "
      asciiwrite(diro+ftxt, sdata(:,:,np))
end do


